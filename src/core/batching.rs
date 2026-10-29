use std::collections::HashMap;

use numpy::{PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::core::types::{ArrayData, GenericObs};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObsDtype {
    Float32,
    Int32,
    Bool,
}

impl ObsDtype {
    pub fn from_str(dtype: &str) -> Option<Self> {
        match dtype.to_ascii_lowercase().as_str() {
            "float32" | "f32" => Some(ObsDtype::Float32),
            "int32" | "i32" => Some(ObsDtype::Int32),
            "bool" | "boolean" => Some(ObsDtype::Bool),
            _ => None,
        }
    }

    pub fn matches(&self, data: &ArrayData) -> bool {
        matches!(
            (self, data),
            (ObsDtype::Float32, ArrayData::Float32(_))
                | (ObsDtype::Int32, ArrayData::Int32(_))
                | (ObsDtype::Bool, ArrayData::Bool(_))
        )
    }
}

#[derive(Clone, Debug)]
pub struct ObsSpecEntry {
    pub shape: Vec<usize>,
    pub dtype: ObsDtype,
}

enum PreparedColumn {
    Float32 { data: Vec<f32>, elem_len: usize },
    Int32 { data: Vec<i32>, elem_len: usize },
    Bool { data: Vec<bool>, elem_len: usize },
}

fn build_full_shape(
    obs_spec: &HashMap<String, ObsSpecEntry>,
    key: &str,
    element_len: usize,
    batch: usize,
) -> Vec<usize> {
    let mut full_shape = Vec::new();
    full_shape.push(batch);
    if let Some(spec) = obs_spec.get(key) {
        full_shape.extend(spec.shape.iter().copied());
    } else {
        full_shape.push(element_len);
    }
    full_shape
}

pub fn stack_action_masks<'py>(
    py: Python<'py>,
    masks: &[Vec<bool>],
) -> PyResult<Option<Py<PyAny>>> {
    if masks.is_empty() {
        return Ok(None);
    }

    let mask_len = masks[0].len();
    for (idx, mask) in masks.iter().enumerate() {
        if mask.len() != mask_len {
            return Err(PyValueError::new_err(format!(
                "Action mask length mismatch at index {} (expected {}, got {})",
                idx,
                mask_len,
                mask.len()
            )));
        }
    }

    let mut flat_masks = Vec::with_capacity(mask_len * masks.len());
    for mask in masks {
        flat_masks.extend_from_slice(mask);
    }

    let stacked = flat_masks.to_pyarray(py).reshape((masks.len(), mask_len))?;
    Ok(Some(stacked.into_any().unbind()))
}

/// Stack rows that are all exactly `width` long (the caller validates that) into an
/// `[n, width]` bool array. Unlike [`stack_action_masks`] it also answers `[0, width]` for
/// an empty batch.
pub fn stack_fixed_width_masks<'py>(
    py: Python<'py>,
    masks: &[Vec<bool>],
    width: usize,
) -> PyResult<Py<PyAny>> {
    let mut flat_masks = Vec::with_capacity(width * masks.len());
    for mask in masks {
        flat_masks.extend_from_slice(mask);
    }
    let stacked = flat_masks.to_pyarray(py).reshape((masks.len(), width))?;
    Ok(stacked.into_any().unbind())
}

/// Below this many stacked elements (all keys, whole batch) `stack_observations` stays on the
/// calling thread.
const PARALLEL_STACK_MIN_ELEMS: usize = 1 << 18;

pub fn stack_observations<'py>(
    py: Python<'py>,
    obs_spec: &HashMap<String, ObsSpecEntry>,
    obs_key_names: &[String],
    obs_list: &[GenericObs],
) -> PyResult<Py<PyAny>> {
    if obs_list.is_empty() {
        // A zero-row batch (e.g. simulate_batch([], [])): empty arrays of the spec's shapes.
        let dict = PyDict::new(py);
        for key in obs_key_names {
            let entry = &obs_spec[key];
            let shape: Vec<usize> = std::iter::once(0)
                .chain(entry.shape.iter().copied())
                .collect();
            let empty = match entry.dtype {
                ObsDtype::Float32 => Vec::<f32>::new().to_pyarray(py).reshape(shape)?.into_any(),
                ObsDtype::Int32 => Vec::<i32>::new().to_pyarray(py).reshape(shape)?.into_any(),
                ObsDtype::Bool => Vec::<bool>::new().to_pyarray(py).reshape(shape)?.into_any(),
            };
            dict.set_item(key, empty)?;
        }
        return Ok(dict.into_any().unbind());
    }

    let batch = obs_list.len();
    let key_count = obs_key_names.len();

    for (idx, obs) in obs_list.iter().enumerate() {
        if obs.len() != key_count {
            return Err(PyValueError::new_err(format!(
                "Observation length mismatch at index {} (expected {} keys, got {})",
                idx,
                key_count,
                obs.len()
            )));
        }
    }

    // Stack each key into one contiguous column. Columns go to rayon's pool only when there is
    // enough to copy: handing a small batch to the pool costs two cross-thread wake-ups per
    // `recv`, more than the copy itself.
    let total_elems: usize = obs_list[0].iter().map(|a| a.len()).sum::<usize>() * batch;
    let parallel = key_count > 1 && total_elems >= PARALLEL_STACK_MIN_ELEMS;
    let build_column = |key_idx: usize| {
        let key = &obs_key_names[key_idx];
        let first = obs_list
            .get(0)
            .and_then(|obs| obs.get(key_idx))
            .ok_or_else(|| format!("Missing observation at index {} for key '{}'", key_idx, key))?;

        if let Some(spec) = obs_spec.get(key) {
            if !spec.dtype.matches(first) {
                return Err(format!("Observation dtype mismatch for key '{}'", key));
            }
        }

        match first {
            ArrayData::Float32(v0) => {
                let mut buffer = Vec::with_capacity(v0.len() * batch);
                for (obs_idx, obs) in obs_list.iter().enumerate() {
                    match obs.get(key_idx) {
                        Some(ArrayData::Float32(v)) => {
                            if v.len() != v0.len() {
                                return Err(format!(
                                                "Observation length mismatch for key '{}' at index {} (expected {}, got {})",
                                                key,
                                                obs_idx,
                                                v0.len(),
                                                v.len()
                                            ));
                            }
                            buffer.extend_from_slice(v);
                        }
                        Some(_) => {
                            return Err(format!(
                                "Observation key '{}' has inconsistent types across batch",
                                key
                            ));
                        }
                        None => {
                            return Err(format!(
                                "Observation at index {} missing key index {}",
                                obs_idx, key_idx
                            ));
                        }
                    }
                }
                Ok(PreparedColumn::Float32 {
                    data: buffer,
                    elem_len: v0.len(),
                })
            }
            ArrayData::Int32(v0) => {
                let mut buffer = Vec::with_capacity(v0.len() * batch);
                for (obs_idx, obs) in obs_list.iter().enumerate() {
                    match obs.get(key_idx) {
                        Some(ArrayData::Int32(v)) => {
                            if v.len() != v0.len() {
                                return Err(format!(
                                                "Observation length mismatch for key '{}' at index {} (expected {}, got {})",
                                                key,
                                                obs_idx,
                                                v0.len(),
                                                v.len()
                                            ));
                            }
                            buffer.extend_from_slice(v);
                        }
                        Some(_) => {
                            return Err(format!(
                                "Observation key '{}' has inconsistent types across batch",
                                key
                            ));
                        }
                        None => {
                            return Err(format!(
                                "Observation at index {} missing key index {}",
                                obs_idx, key_idx
                            ));
                        }
                    }
                }
                Ok(PreparedColumn::Int32 {
                    data: buffer,
                    elem_len: v0.len(),
                })
            }
            ArrayData::Bool(v0) => {
                let mut buffer = Vec::with_capacity(v0.len() * batch);
                for (obs_idx, obs) in obs_list.iter().enumerate() {
                    match obs.get(key_idx) {
                        Some(ArrayData::Bool(v)) => {
                            if v.len() != v0.len() {
                                return Err(format!(
                                                "Observation length mismatch for key '{}' at index {} (expected {}, got {})",
                                                key,
                                                obs_idx,
                                                v0.len(),
                                                v.len()
                                            ));
                            }
                            buffer.extend_from_slice(v);
                        }
                        Some(_) => {
                            return Err(format!(
                                "Observation key '{}' has inconsistent types across batch",
                                key
                            ));
                        }
                        None => {
                            return Err(format!(
                                "Observation at index {} missing key index {}",
                                obs_idx, key_idx
                            ));
                        }
                    }
                }
                Ok(PreparedColumn::Bool {
                    data: buffer,
                    elem_len: v0.len(),
                })
            }
        }
    };
    let stacked_columns = if parallel {
        py.detach(|| {
            (0..key_count)
                .into_par_iter()
                .map(build_column)
                .collect::<Result<Vec<_>, String>>()
        })
    } else {
        (0..key_count)
            .map(build_column)
            .collect::<Result<Vec<_>, String>>()
    }
    .map_err(PyValueError::new_err)?;

    let obs_dict = PyDict::new(py);

    for (key_idx, column) in stacked_columns.into_iter().enumerate() {
        let key = &obs_key_names[key_idx];
        match column {
            PreparedColumn::Float32 { data, elem_len } => {
                let full_shape = build_full_shape(obs_spec, key, elem_len, batch);
                let reshaped = data.to_pyarray(py).reshape(full_shape.as_slice())?;
                obs_dict.set_item(key, reshaped)?;
            }
            PreparedColumn::Int32 { data, elem_len } => {
                let full_shape = build_full_shape(obs_spec, key, elem_len, batch);
                let reshaped = data.to_pyarray(py).reshape(full_shape.as_slice())?;
                obs_dict.set_item(key, reshaped)?;
            }
            PreparedColumn::Bool { data, elem_len } => {
                let full_shape = build_full_shape(obs_spec, key, elem_len, batch);
                let reshaped = data.to_pyarray(py).reshape(full_shape.as_slice())?;
                obs_dict.set_item(key, reshaped)?;
            }
        }
    }

    Ok(obs_dict.into_any().unbind())
}
