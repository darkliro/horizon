use super::sprite_split;

/// CPU texture uploads prepared by the resource worker. Consumed by renderer
/// construction so the temporary mip chains are released after GPU upload.
pub struct PreparedTextures {
    pub(super) sprites: Vec<(Vec<u8>, u32)>,
    /// The sprite layers per array `sprites` is cut by.
    pub(super) split: u32,
    pub(super) forcefield: (Vec<u8>, u32),
    /// The LabPBR maps' mip chains (normals, specular), their levels, layers and size.
    pub(super) pbr: Option<(Vec<u8>, Vec<u8>, u32, u32, u32)>,
}

/// Every level of `layers` square RGBA layers of `size`, each the mean of four texels of the
/// one above, layer major: the LabPBR maps' mips (their channels are data, not colour).
fn box_mip_chain(pixels: &[u8], size: u32, layers: u32) -> Vec<u8> {
    let layer_bytes = (size * size * 4) as usize;
    let mut out = Vec::new();
    for layer in pixels.chunks_exact(layer_bytes).take(layers as usize) {
        let mut level = layer.to_vec();
        let mut side = size as usize;
        out.extend_from_slice(&level);
        while side > 1 {
            let half = side / 2;
            let mut next = Vec::with_capacity(half * half * 4);
            for y in 0..half {
                for x in 0..half {
                    for c in 0..4 {
                        let at = |dx: usize, dy: usize| {
                            u32::from(level[((2 * y + dy) * side + 2 * x + dx) * 4 + c])
                        };
                        next.push(((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1) + 2) / 4) as u8);
                    }
                }
            }
            out.extend_from_slice(&next);
            level = next;
            side = half;
        }
    }
    out
}

impl PreparedTextures {
    pub fn new(
        pixels: &[u8],
        size: u32,
        layers: u32,
        settings: &[crate::mipmap::Settings],
        forcefield: (&[u8], u32),
    ) -> Self {
        Self::with_split(pixels, size, layers, settings, forcefield, sprite_split())
    }

    pub(crate) fn with_split(
        pixels: &[u8],
        size: u32,
        layers: u32,
        settings: &[crate::mipmap::Settings],
        forcefield: (&[u8], u32),
        split: u32,
    ) -> Self {
        let layer_bytes = (size * size * 4) as usize;
        Self {
            sprites: (0..layers.div_ceil(split).max(1))
                .map(|array| {
                    let first = array * split;
                    let count = layers.saturating_sub(first).clamp(1, split);
                    let start = (first as usize * layer_bytes).min(pixels.len());
                    mip_chain(&pixels[start..], size, settings, first, count)
                })
                .collect(),
            split,
            forcefield: mip_chain(forcefield.0, forcefield.1, &[], 0, 1),
            pbr: None,
        }
    }

    /// With the resource packs' LabPBR `normals` and `specular` maps (`layers` layers of
    /// `size`, as `RenderAssets` holds them; nothing when empty).
    pub fn with_pbr(mut self, normals: &[u8], specular: &[u8], size: u32, layers: u32) -> Self {
        if !normals.is_empty() {
            self.pbr = Some((
                box_mip_chain(normals, size, layers),
                box_mip_chain(specular, size, layers),
                size.trailing_zeros() + 1,
                layers,
                size,
            ));
        }
        self
    }

    /// The same mip chains cut by `split` layers per array: chains made before
    /// the adapter was known, for a GPU with fewer layers per array.
    pub(crate) fn resplit(mut self, layers: u32, split: u32) -> Self {
        if self.split == split || layers <= split.min(self.split) {
            self.split = split;
            return self;
        }
        let levels = self.sprites[0].1;
        let all: Vec<u8> = self
            .sprites
            .drain(..)
            .flat_map(|(bytes, _)| bytes)
            .collect();
        // Every layer's mip levels are contiguous (layer major).
        let layer_bytes = all.len() / layers as usize;
        self.sprites = all
            .chunks(layer_bytes * split as usize)
            .map(|chunk| (chunk.to_vec(), levels))
            .collect();
        self.split = split;
        self
    }
}

pub(super) fn mip_chain(
    pixels: &[u8],
    size: u32,
    settings: &[crate::mipmap::Settings],
    first: u32,
    layers: u32,
) -> (Vec<u8>, u32) {
    let levels = size.trailing_zeros() + 1;
    let layer_bytes = (size * size * 4) as usize;
    // Exactly the chain's size: cached animation frames count their capacity.
    let chain_bytes: usize = (0..levels).map(|l| layer_bytes >> (2 * l)).sum();
    let mut out = Vec::with_capacity(chain_bytes * layers as usize);
    // Every layer makes its own mips from its own pixels, and a pack's array holds
    // thousands of them: the layers run on every thread, in chunks so a worker
    // allocates one chain per chunk rather than one per layer. The chunks are
    // appended in order, so the chain stays layer major.
    const LAYERS_PER_CHUNK: u32 = 16;
    let chunks: Vec<(u32, u32)> = (0..layers)
        .step_by(LAYERS_PER_CHUNK as usize)
        .map(|start| (start, (layers - start).min(LAYERS_PER_CHUNK)))
        .collect();
    let parts = crate::par::par_map(&chunks, |&(start, count)| {
        mip_chain_layers(
            &pixels[start as usize * layer_bytes..],
            size,
            settings,
            first + start,
            count,
            levels - 1,
            layer_bytes,
        )
    });
    for part in parts {
        out.extend_from_slice(&part);
    }
    debug_assert_eq!(out.len(), chain_bytes * layers as usize);
    (out, levels)
}

/// The mip chain of `layers` layers from `pixels` that start at layer `first`:
/// `layers`' base levels followed by their reductions, layer major.
fn mip_chain_layers(
    pixels: &[u8],
    size: u32,
    settings: &[crate::mipmap::Settings],
    first: u32,
    layers: u32,
    levels: u32,
    layer_bytes: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        (0..=levels).map(|l| layer_bytes >> (2 * l)).sum::<usize>() * layers as usize,
    );
    for layer in 0..layers as usize {
        let base = crate::mipmap::Image {
            width: size,
            height: size,
            pixels: pixels[layer * layer_bytes..(layer + 1) * layer_bytes]
                .as_chunks::<4>()
                .0
                .to_vec(),
        };
        let s = settings
            .get(first as usize + layer)
            .copied()
            .unwrap_or_default();
        let has_transparent = s.has_transparent || base.pixels.iter().any(|p| p[3] == 0);
        for image in crate::mipmap::generate(
            base,
            levels,
            crate::mipmap::Settings {
                has_transparent,
                ..s
            },
        ) {
            out.extend(image.pixels.iter().flatten());
        }
    }
    out
}
