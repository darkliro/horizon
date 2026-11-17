//! GPU error scopes (out-of-memory and validation).

/// Out-of-memory and validation error scopes, popped together: the first error either caught.
pub(super) struct GpuChecks {
    memory: wgpu::ErrorScopeGuard,
    validation: wgpu::ErrorScopeGuard,
}

impl GpuChecks {
    pub(super) fn push(device: &wgpu::Device) -> Self {
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        Self { memory, validation }
    }

    pub(super) fn pop(self) -> Option<wgpu::Error> {
        let validation = pollster::block_on(self.validation.pop());
        let memory = pollster::block_on(self.memory.pop());
        memory.or(validation)
    }
}
