use super::*;

/// entity.wgsl after the OverlayTexture constants it reads.
pub(super) fn entity_shader() -> String {
    crate::overlay::wgsl_constants() + wgsl_prefix::HDR_LIGHT + include_str!("../entity.wgsl")
}

pub(super) fn entity_binding_entries() -> [wgpu::BindGroupLayoutEntry; 8] {
    let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
    let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: None,
    };
    [
        entry(
            0,
            vf,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        ),
        entry(
            1,
            vf,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        ),
        entry(
            2,
            vf,
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        ),
        entry(
            3,
            vf,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        ),
        entry(
            4,
            vf,
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        ),
        entry(
            5,
            vf,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        ),
        entry(
            6,
            vf,
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        ),
        entry(
            7,
            vf,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        ),
    ]
}

pub(super) const ENTITY_ATTRIBUTES: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x2,2=>Uint32,3=>Unorm8x4,4=>Uint8x4,5=>Snorm8x4,6=>Float32x2];

pub(super) struct EntityPipelineOptions<'a> {
    pub(super) label: &'a str,
    pub(super) cull_mode: Option<wgpu::Face>,
    pub(super) depth_write: bool,
    pub(super) compare: wgpu::CompareFunction,
    pub(super) bias: wgpu::DepthBiasState,
    pub(super) blend: wgpu::BlendState,
    pub(super) clouds: bool,
    /// `vs_entity` for the entity vertex layout. Clouds still enter at `vs_cloud`.
    pub(super) vertex: &'a str,
    pub(super) fragment: &'a str,
    pub(super) write_mask: wgpu::ColorWrites,
}

/// Constructs the entity shader pipelines shared by the renderer and its pixel tests.
pub(super) fn entity_pipeline_factory<'a>(
    kept: &'a reuse::On<'a>,
    entity_layout: &'a wgpu::PipelineLayout,
    cloud_layout: &'a wgpu::PipelineLayout,
    entity_shader: &'a wgpu::ShaderModule,
    entity_vertex: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    frame_format: wgpu::TextureFormat,
    linear: bool,
) -> impl Fn(EntityPipelineOptions<'_>) -> wgpu::RenderPipeline + Copy + 'a {
    move |options| {
        let EntityPipelineOptions {
            label,
            cull_mode,
            depth_write,
            compare,
            bias,
            blend,
            clouds,
            vertex,
            fragment,
            write_mask,
        } = options;
        kept.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(if clouds { cloud_layout } else { entity_layout }),
            vertex: wgpu::VertexState {
                module: entity_shader,
                entry_point: Some(if clouds { "vs_cloud" } else { vertex }),
                compilation_options: Default::default(),
                buffers: entity_vertex,
            },
            primitive: wgpu::PrimitiveState {
                cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(depth_write),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias,
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: entity_shader,
                entry_point: Some(fragment),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &linear_blend::frame_constants(frame_format, linear),
                    ..Default::default()
                },
                targets: &[Some(wgpu::ColorTargetState {
                    format: frame_format,
                    blend: Some(blend),
                    write_mask,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    }
}

/// The static block-entity vertex entry point shares entity shading but lives
/// in its own source file and pipeline.
pub(super) fn static_entity_shader() -> String {
    entity_shader() + include_str!("../static_entity.wgsl")
}

impl Renderer {
    /// Uploads one generation of prepared geometry. A failed upload leaves the previous
    /// generation in place.
    pub fn install_prepared_models(
        &mut self,
        generation: u64,
        vertices: &[PreparedModelVertex],
        ranges: Vec<Option<(u32, u32)>>,
    ) -> Result<()> {
        if vertices.is_empty() {
            self.prepared_models = None;
            return Ok(());
        }
        let next = PreparedModels::create(
            &self.device,
            &self.reuse,
            &self.entity_bindings,
            prepared_model::prepared_attachment(self.output.frame.format),
            generation,
            vertices,
            ranges,
        )?;
        self.prepared_models = Some(next);
        Ok(())
    }

    /// The uploaded generation, when a prepared buffer is resident.
    pub fn prepared_generation(&self) -> Option<u64> {
        self.prepared_models
            .as_ref()
            .map(|models| models.generation())
    }

    /// This frame's prepared copies, in entity-stream order. The stored list keeps its capacity.
    pub fn stage_prepared_draws(&mut self, draws: &[PreparedModelDraw]) {
        self.prepared_draws.clear();
        self.prepared_draws.extend_from_slice(draws);
    }

    /// How many of `incoming` copies still fit after `already` are staged.
    pub fn prepared_instance_room(&self, already: u64, incoming: usize) -> usize {
        prepared_model::prepared_instance_room_for(
            already,
            incoming,
            self.device.limits().max_buffer_size,
        )
    }

    pub(super) fn finish_prepared_frame(&mut self) {
        prepared_model::finish_staged(&mut self.prepared_draws);
        self.prepared_cuts.clear();
        self.prepared_ranges.clear();
    }

    pub(super) fn draw_hand(&self, pass: &mut wgpu::RenderPass<'_>, quads: u32, first: u32) {
        pass.set_pipeline(&self.entity_pipeline);
        pass.set_bind_group(0, &self.hand_group, &[]);
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_vertex_buffer(0, self.entity_buffer.slice(..));
        pass.draw_indexed(0..quads * 6, (first * 4) as i32, 0..1);
    }
    /// LevelRenderer.submitBlockOutline: the buffer holds `edges` per stroke, the black
    /// secondary stroke first when high contrast is on.
    #[horizon_mod_api_macros::replaceable]
    pub(super) fn draw_outline(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        edges: u32,
        high_contrast: bool,
    ) {
        pass.set_vertex_buffer(0, self.outline.slice(..));
        // Both strokes are LINES: a pack's rendertype_lines draws them.
        if let Some(lines) = &self.core_lines {
            if high_contrast {
                lines.draw(pass, true, 0..edges);
            }
            let first = if high_contrast { edges } else { 0 };
            lines.draw(pass, false, first..first + edges);
            pass.set_bind_group(0, &self.group, &[]);
            return;
        }
        if high_contrast {
            pass.set_pipeline(&self.lines_secondary);
            pass.draw(0..4, 0..edges);
        }
        pass.set_pipeline(&self.lines);
        let first = if high_contrast { edges } else { 0 };
        pass.draw(0..4, first..first + edges);
    }
}
