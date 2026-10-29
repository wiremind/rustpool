#[derive(Debug, Clone)]
pub enum ArrayData {
    Float32(Vec<f32>),
    Int32(Vec<i32>),
    Bool(Vec<bool>),
}

impl ArrayData {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            ArrayData::Float32(v) => v.len(),
            ArrayData::Int32(v) => v.len(),
            ArrayData::Bool(v) => v.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub type GenericObs = Vec<ArrayData>;
