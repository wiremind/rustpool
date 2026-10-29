# Python API

Generated from the docstrings of the installed package (the pool's from the compiled module).

## Making a pool

::: rustpool.make

::: rustpool.list_envs

## The pool

`rustpool.RlPool` is the compiled pool class (`rustpool.MyEnvPool`, its old name, is a
deprecated alias). A pool built by
[your own crate](guides/own-crate.md) has the same methods.

::: rustpool._rustpool.RlPool
    options:
      force_inspection: true
      show_root_full_path: false
      members_order: source
      filters: ["!^_"]
      separate_signature: true

## JAX bridge

::: rustpool.jax_api.JaxPool
    options:
      members_order: source
      filters: ["!^_"]

::: rustpool.jax_api.EnvOutput
    options:
      show_bases: false

## PyTorch

::: rustpool.torch_api.TorchEnvPool
    options:
      members_order: source
      filters: ["!^_"]

::: rustpool.torch_api.to_torch

## gymnasium wrapper

A `gymnasium.vector.VectorEnv` over a pool. Needs the `gymnasium` extra
(`pip install "rustpool[gymnasium]"`); see [gymnasium](concepts/gymnasium.md).

::: rustpool.envpool_api.gymnasium.RustPoolGymnasiumWrapper
    options:
      members_order: source
      filters: ["!^_"]
