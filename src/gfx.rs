// Steps before drawing something to the screen.
// 1) Create Instance
// 2) Create Surface
// 3) Select a Physical Device (or multiple)
// 4) Create a logical device and get queues
// 5) Create pools
// 6) Create a Render Pass
// 7) Create a Swapchain and get images

use crate::constants::{ APP_NAME, MAX_FRAMES_IN_FLIGHT };
use crate::vertex::{Vertex, SQARE_PRIMITIVE};
use gfx_backend_metal as backend;
use gfx_hal as hal;
use winit;

use gfx_hal::adapter::PhysicalDevice;
use self::hal::{
    adapter, memory,
    command, format, image, pass, pool, pso, queue, window, Adapter, Instance,
    Backbuffer, Backend, Device, Graphics, Primitive, Surface, Swapchain,
    SwapchainConfig,
};

pub struct HalState {
    pub in_flight_fences: Vec<<backend::Backend as Backend>::Fence>,
    pub render_finished_semaphores: Vec<<backend::Backend as Backend>::Semaphore>,
    pub image_available_semaphores: Vec<<backend::Backend as Backend>::Semaphore>,
    pub submission_command_buffers: Vec<
        command::Submit<
            backend::Backend,
            Graphics,
            command::MultiShot,
            command::Primary,
        >,
    >,
    pub command_pool: pool::CommandPool<backend::Backend, Graphics>,
    pub swapchain_framebuffers: Vec<<backend::Backend as Backend>::Framebuffer>,
    pub gfx_pipeline: <backend::Backend as Backend>::GraphicsPipeline,
    pub descriptor_set_layouts:
        Vec<<backend::Backend as Backend>::DescriptorSetLayout>,
    pub pipeline_layout: <backend::Backend as Backend>::PipelineLayout,
    pub render_pass: <backend::Backend as Backend>::RenderPass,
    pub frame_images: Vec<(
        <backend::Backend as Backend>::Image,
        <backend::Backend as Backend>::ImageView,
    )>,
    pub _format: format::Format,
    pub swapchain: <backend::Backend as Backend>::Swapchain,
    pub command_queues: Vec<queue::CommandQueue<backend::Backend, Graphics>>,
    pub device: <backend::Backend as Backend>::Device,
    pub _surface: <backend::Backend as Backend>::Surface,
    pub _adapter: Adapter<backend::Backend>,
    pub _instance: backend::Instance,
}

impl HalState {
    pub fn create_instance() -> backend::Instance {
        backend::Instance::create(APP_NAME, 1)
    }

    pub fn create_surface(
        instance: &backend::Instance,
        window: &winit::Window,
    ) -> backend::Surface {
        instance.create_surface(window)
    }

    /// An adapter is a bridge between our graphic card and the graphic API.
    /// This method will return the first adapter found.
    pub fn pick_adapter(
        instance: &backend::Instance,
    ) -> hal::Adapter<backend::Backend> {
        instance
            .enumerate_adapters()
            .into_iter()
            .next()
            .expect("No adapter found")
    }

    pub fn create_device_with_graphics_queues(
        adapter: &mut hal::Adapter<backend::Backend>,
        surface: &backend::Surface,
    ) -> (
        backend::Device,
        hal::queue::family::QueueGroup<backend::Backend, Graphics>,
	Vec<queue::CommandQueue<backend::Backend, Graphics>>
    ) {
        let num_queues = 1;
        let (device, mut queue_group) = adapter
            .open_with::<_, Graphics>(num_queues, |family| {
                surface.supports_queue_family(family)
            })
            .expect("Could not find a queue family supporting graphics.");

        let command_queues: Vec<_> = queue_group.queues.drain(..1).collect();

        (device, queue_group, command_queues)
    }

    pub fn create_swapchain(
        adapter: &Adapter<backend::Backend>,
        device: &backend::Device,
        surface: &mut backend::Surface,
        previous_swapchain: Option<backend::Swapchain>,
    ) -> (
        backend::Swapchain,
        hal::window::Extent2D,
        Backbuffer<backend::Backend>,
        format::Format,
    ) {
        let (caps, formats, _present_modes) =
            surface.compatibility(&adapter.physical_device);

        let format = formats.map_or(format::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| {
                    format.base_format().1 == format::ChannelType::Srgb
                })
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let swap_config = SwapchainConfig::from_caps(&caps, format);
        let extent = swap_config.extent;
        let (swapchain, backbuffer) =
            device.create_swapchain(surface, swap_config, previous_swapchain);

        (swapchain, extent, backbuffer, format)
    }

    pub fn create_image_views(
        backbuffer: Backbuffer<backend::Backend>,
        format: format::Format,
        device: &<backend::Backend as Backend>::Device,
    ) -> Vec<(
        <backend::Backend as Backend>::Image,
        <backend::Backend as Backend>::ImageView,
    )> {
        match backbuffer {
            window::Backbuffer::Images(images) => images
                .into_iter()
                .map(|image| {
                    let image_view = match device.create_image_view(
                        &image,
                        image::ViewKind::D2,
                        format,
                        format::Swizzle::NO,
                        image::SubresourceRange {
                            aspects: format::Aspects::COLOR,
                            levels: 0..1,
                            layers: 0..1,
                        },
                    ) {
                        Ok(image_view) => image_view,
                        Err(_) => {
                            panic!("Error creating image view for an image!")
                        }
                    };

                    (image, image_view)
                })
                .collect(),
            _ => unimplemented!(),
        }
    }

    pub fn cleanup(self) {
        // Cleanup
        self.device.destroy_graphics_pipeline(self.gfx_pipeline);
        self.device.destroy_pipeline_layout(self.pipeline_layout);
        self.device.destroy_render_pass(self.render_pass);
        self.device.destroy_command_pool(self.command_pool.into_raw());

        for fence in self.in_flight_fences {
            self.device.destroy_fence(fence);
        };

        for semaphore in self.image_available_semaphores{
            self.device.destroy_semaphore(semaphore);
        };
    }

    pub fn create_framebuffers(
        device: &<backend::Backend as hal::Backend>::Device,
        render_pass: &<backend::Backend as hal::Backend>::RenderPass,
        frame_images: &[(
            <backend::Backend as hal::Backend>::Image,
            <backend::Backend as hal::Backend>::ImageView,
        )],
        extent: window::Extent2D,
    ) -> Vec<<backend::Backend as hal::Backend>::Framebuffer> {
        let mut swapchain_framebuffers: Vec<
            <backend::Backend as hal::Backend>::Framebuffer,
        > = Vec::new();

        for (_, image_view) in frame_images.iter() {
            swapchain_framebuffers.push(
                device
                    .create_framebuffer(
                        render_pass,
                        vec![image_view],
                        image::Extent {
                            width: extent.width as _,
                            height: extent.height as _,
                            depth: 1,
                        },
                    )
                    .expect("failed to create framebuffer!"),
            );
        }

        swapchain_framebuffers
    }

    pub fn create_render_pass(
        device: &backend::Device,
        format: Option<format::Format>,
    ) -> <backend::Backend as hal::Backend>::RenderPass {
        let samples = 1;

        let ops = pass::AttachmentOps::new(
            pass::AttachmentLoadOp::Clear,
            pass::AttachmentStoreOp::Store,
        );

        let stencil_ops = pass::AttachmentOps::DONT_CARE;
        let layouts = image::Layout::Undefined..image::Layout::Present;

        let color_attachment = pass::Attachment {
            format,
            samples,
            ops,
            stencil_ops,
            layouts,
        };

        let subpass = pass::SubpassDesc {
            colors: &[(0, image::Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let render_pass =
            device.create_render_pass(&[color_attachment], &[subpass], &[]);

        render_pass
    }

    pub fn create_command_buffers<'a>(
        command_pool: &'a mut hal::pool::CommandPool<
            backend::Backend,
            Graphics,
        >,
        render_pass: &<backend::Backend as hal::Backend>::RenderPass,
        framebuffers: &[<backend::Backend as hal::Backend>::Framebuffer],
        extent: window::Extent2D,
        pipeline: &<backend::Backend as hal::Backend>::GraphicsPipeline,
        vertex_buffer: <backend::Backend as Backend>::Buffer,
    ) -> Vec<
        command::Submit<
            backend::Backend,
            Graphics,
            command::MultiShot,
            command::Primary,
        >,
    > {
        // reserve (allocate memory for) primary command buffers
        command_pool.reserve(framebuffers.iter().count());

        let mut submission_command_buffers: Vec<
            command::Submit<
                backend::Backend,
                Graphics,
                command::MultiShot,
                command::Primary,
            >,
        > = Vec::new();

        for fb in framebuffers.iter() {
            let mut command_buffer: command::CommandBuffer<
                backend::Backend,
                Graphics,
                command::MultiShot,
                command::Primary,
            > = command_pool.acquire_command_buffer(false);

            command_buffer.bind_graphics_pipeline(pipeline);
            command_buffer.bind_vertex_buffers(0, vec![(&vertex_buffer, 0)]);

            {
                // begin render pass
                let render_area = pso::Rect {
                    x: 0,
                    y: 0,
                    w: extent.width as _,
                    h: extent.height as _,
                };
                let clear_values = vec![command::ClearValue::Color(
                    command::ClearColor::Float([0.0, 0.0, 0.0, 1.0]),
                )];

                let mut render_pass_inline_encoder = command_buffer
                    .begin_render_pass_inline(
                        render_pass,
                        fb,
                        render_area,
                        clear_values.iter(),
                    );

                let num_vertices = SQARE_PRIMITIVE.len() as u32;
                render_pass_inline_encoder.draw(0..num_vertices, 0..1);
            }

            let submission_command_buffer = command_buffer.finish();
            submission_command_buffers.push(submission_command_buffer);
        }

        submission_command_buffers
    }

    pub fn create_command_pool(
        device: &<backend::Backend as hal::Backend>::Device,
        qf_id: queue::family::QueueFamilyId,
    ) -> pool::CommandPool<backend::Backend, Graphics> {
        let raw_command_pool = device
            .create_command_pool(qf_id, pool::CommandPoolCreateFlags::empty());

        unsafe { 
            pool::CommandPool::new(raw_command_pool)
        }
    }

    pub fn draw(&mut self, current_frame: usize) {
        Self::draw_frame(
            &self.device,
            &mut self.command_queues,
            &mut self.swapchain,
            &self.submission_command_buffers,
            &mut self.image_available_semaphores[current_frame],
            &mut self.render_finished_semaphores[current_frame],
            &mut self.in_flight_fences[current_frame],
        );
    }

    fn draw_frame(
        device: &<backend::Backend as hal::Backend>::Device,
        command_queues: &mut [hal::queue::CommandQueue<
            backend::Backend,
            Graphics,
        >],
        swapchain: &mut <backend::Backend as hal::Backend>::Swapchain,
        submission_command_buffers: &[hal::command::Submit<
            backend::Backend,
            Graphics,
            command::MultiShot,
            command::Primary,
        >],
        image_available_semaphore: &<backend::Backend as hal::Backend>::Semaphore,
        render_finished_semaphore: &<backend::Backend as hal::Backend>::Semaphore,
        in_flight_fence: &<backend::Backend as hal::Backend>::Fence,
    ) {
        device.wait_for_fence(in_flight_fence, std::u64::MAX);
        device.reset_fence(in_flight_fence);

        let image_index = swapchain
            .acquire_image(
                std::u64::MAX,
                window::FrameSync::Semaphore(image_available_semaphore),
            )
            .expect("could not acquire image!");

        let submission = queue::submission::Submission::new()
            .wait_on(&[(
                image_available_semaphore,
                pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            )])
            .signal(vec![render_finished_semaphore])
            .submit(Some(&submission_command_buffers[image_index as usize]));

        // recall we only made one queue
        command_queues[0].submit(submission, Some(in_flight_fence));

        swapchain
            .present(
                &mut command_queues[0],
                image_index,
                vec![render_finished_semaphore],
            )
            .expect("presentation failed!");
    }

    pub fn create_graphics_pipeline(
        device: &<backend::Backend as Backend>::Device,
        adapter: &hal::Adapter<backend::Backend>,
        extent: window::Extent2D,
        render_pass: &<backend::Backend as Backend>::RenderPass,
    ) -> (
        Vec<<backend::Backend as Backend>::DescriptorSetLayout>,
        <backend::Backend as Backend>::PipelineLayout,
        <backend::Backend as Backend>::GraphicsPipeline,
        <backend::Backend as Backend>::Buffer,
        <backend::Backend as Backend>::Memory,
    ) {

        let vert_shader_code = include_bytes!("../assets/shaders/default.vert.spv");
        let frag_shader_code = include_bytes!("../assets/shaders/default.frag.spv");

        let vert_shader_module = device
            .create_shader_module(vert_shader_code)
            .expect("Error creating shader module.");
        let frag_shader_module = device
            .create_shader_module(frag_shader_code)
            .expect("Error creating fragment module.");

        let specialization = pso::Specialization::default();

            let (vs_entry, fs_entry) = (
                pso::EntryPoint::<backend::Backend> {
                    entry: "main",
                    module: &vert_shader_module,
                    specialization,
                },
                pso::EntryPoint::<backend::Backend> {
                    entry: "main",
                    module: &frag_shader_module,
                    specialization,
                },
            );
            let shaders = pso::GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            let rasterizer = pso::Rasterizer {
                depth_clamping: false,
                polygon_mode: pso::PolygonMode::Fill,
                cull_face: <pso::Face>::BACK,
                front_face: pso::FrontFace::Clockwise,
                depth_bias: None,
                conservative: false,
            };
            let bindings = Vec::<pso::DescriptorSetLayoutBinding>::new();
            let immutable_samplers =
                Vec::<<backend::Backend as Backend>::Sampler>::new();

            let descriptor_set_layouts: Vec<<backend::Backend as Backend>::DescriptorSetLayout> =
                vec![device.create_descriptor_set_layout(
                    bindings,
                    immutable_samplers,
                )];

            let push_constants =
                Vec::<(pso::ShaderStageFlags, std::ops::Range<u32>)>::new();

            let layout = device.create_pipeline_layout(
                &descriptor_set_layouts,
                push_constants,
            );

            let subpass = pass::Subpass {
                index: 0,
                main_pass: render_pass,
            };

            let gfx_pipeline = {
                let mut desc = pso::GraphicsPipelineDesc::new(
                    shaders,
                    Primitive::TriangleList,
                    rasterizer,
                    &layout,
                    subpass
                );

                desc.vertex_buffers.push(pso::VertexBufferDesc {
                    binding: 0,
                    stride: std::mem::size_of::<Vertex>() as u32,
                    rate: 0,
                });

                desc.attributes.push(pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: pso::Element {
                        format: format::Format::Rgb32Float,
                        offset: 0,
                    },
                });

                desc.attributes.push(pso::AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: pso::Element {
                        format: format::Format::Rgba32Float,
                        offset: 12,
                    },
                });

                device
                    .create_graphics_pipeline(&desc, None)
                    .expect("failed to create graphics pipeline!")
            };

            device.destroy_shader_module(vert_shader_module);
            device.destroy_shader_module(frag_shader_module);


            let mesh = SQARE_PRIMITIVE;
            // Here's where we create the buffer itself, and the memory to hold it. There's
            // a lot in here, and in future parts we'll extract it to a utility function.
            let (vertex_buffer, vertex_buffer_memory) = {
                // First we create an unbound buffer (e.g, a buffer not currently bound to
                // any memory). We need to work out the size of it in bytes, and declare
                // that we want to use it for vertex data.
                let item_count = mesh.len();
                let stride = std::mem::size_of::<Vertex>() as u64;
                let buffer_len = item_count as u64 * stride;
                let unbound_buffer = device
                    .create_buffer(buffer_len, hal::buffer::Usage::VERTEX)
                    .unwrap();

                // Next, we need the graphics card to tell us what the memory requirements
                // for this buffer are. This includes the size, alignment, and available
                // memory types. We know how big our data is, but we have to store it in
                // a valid way for the device.
                let req = device.get_buffer_requirements(&unbound_buffer);

                // This complicated looking statement filters through memory types to pick
                // one that's appropriate. We call enumerate to give us the ID (the index)
                // of each type, which might look something like this:
                //
                // id   ty
                // ==   ==
                // 0    DEVICE_LOCAL
                // 1    COHERENT | CPU_VISIBLE
                // 2    DEVICE_LOCAL | CPU_VISIBLE
                // 3    DEVICE_LOCAL | CPU_VISIBLE | CPU_CACHED
                //
                // We then want to find the first type that is supported by our memory
                // requirements (e.g, `id` is in the `type_mask` bitfield), and also has
                // the CPU_VISIBLE property (so we can copy vertex data directly into it.)
                let physical_device = &adapter.physical_device;
                let memory_types = physical_device.memory_properties().memory_types;

                let upload_type = memory_types
                    .iter()
                    .enumerate()
                    .find(|(id, ty)| {
                        let type_supported = req.type_mask & (1_u64 << id) != 0;
                        type_supported && ty.properties.contains(memory::Properties::CPU_VISIBLE)
                    }).map(|(id, _ty)| adapter::MemoryTypeId(id))
                .expect("Could not find approprate vertex buffer memory type.");

                // Now that we know the type and size of memory we need, we can allocate it
                // and bind out buffer to it. The `0` there is an offset, which you could
                // use to bind multiple buffers to the same block of memory.
                let buffer_memory = device.allocate_memory(upload_type, req.size).unwrap();
                let buffer = device
                    .bind_buffer_memory(&buffer_memory, 0, unbound_buffer)
                    .unwrap();

                // Finally, we can copy our vertex data into the buffer. To do this we get
                // a writer corresponding to the range of memory we want to write to. This
                // writer essentially memory maps the data and acts as a slice that we can
                // write into. Once we do that, we unmap the memory, and our buffer should
                // now be full.
                {
                    let mut dest = device
                        .acquire_mapping_writer::<Vertex>(&buffer_memory, 0..buffer_len)
                        .unwrap();
                    dest.copy_from_slice(mesh);
                    device.release_mapping_writer(dest);
                }

                (buffer, buffer_memory)
            };

            (descriptor_set_layouts, layout, gfx_pipeline, vertex_buffer, vertex_buffer_memory)


        //     let (vs_entry, fs_entry) = (
        //         pso::EntryPoint::<backend::Backend> {
        //             entry: "main",
        //             module: &vert_shader_module,
        //             specialization,
        //         },
        //         pso::EntryPoint::<backend::Backend> {
        //             entry: "main",
        //             module: &frag_shader_module,
        //             specialization,
        //         },
        //     );

        //     let shaders = pso::GraphicsShaderSet {
        //         vertex: vs_entry,
        //         hull: None,
        //         domain: None,
        //         geometry: None,
        //         fragment: Some(fs_entry),
        //     };

        //     let rasterizer = pso::Rasterizer {
        //         depth_clamping: false,
        //         polygon_mode: pso::PolygonMode::Fill,
        //         cull_face: <pso::Face>::BACK,
        //         front_face: pso::FrontFace::Clockwise,
        //         depth_bias: None,
        //         conservative: false,
        //     };


        //     // let input_assembler =
        //     //     pso::InputAssemblerDesc::new(Primitive::TriangleList);

        //     // let blender = {
        //     //     let blend_state = pso::BlendState::On {
        //     //         color: pso::BlendOp::Add {
        //     //             src: pso::Factor::One,
        //     //             dst: pso::Factor::Zero,
        //     //         },
        //     //         alpha: pso::BlendOp::Add {
        //     //             src: pso::Factor::One,
        //     //             dst: pso::Factor::Zero,
        //     //         },
        //     //     };

        //     //     pso::BlendDesc {
        //     //         logic_op: Some(pso::LogicOp::Copy),
        //     //         targets: vec![pso::ColorBlendDesc(
        //     //             pso::ColorMask::ALL,
        //     //             blend_state,
        //     //         )],
        //     //     }
        //     // };

        //     // let depth_stencil = pso::DepthStencilDesc {
        //     //     depth: pso::DepthTest::Off,
        //     //     depth_bounds: false,
        //     //     stencil: pso::StencilTest::Off,
        //     // };

        //     // let multisampling: Option<pso::Multisampling> = None;

        //     // let baked_states = pso::BakedStates {
        //     //     viewport: Some(pso::Viewport {
        //     //         rect: pso::Rect {
        //     //             x: 0,
        //     //             y: 0,
        //     //             w: extent.width as i16,
        //     //             h: extent.width as i16,
        //     //         },
        //     //         depth: (0.0..1.0),
        //     //     }),
        //     //     scissor: Some(pso::Rect {
        //     //         x: 0,
        //     //         y: 0,
        //     //         w: extent.width as i16,
        //     //         h: extent.height as i16,
        //     //     }),
        //     //     blend_color: None,
        //     //     depth_bounds: None,
        //     // };

        //     let bindings = Vec::<pso::DescriptorSetLayoutBinding>::new();
        //     let immutable_samplers =
        //         Vec::<<backend::Backend as Backend>::Sampler>::new();

        //     let descriptor_set_layouts: Vec<<backend::Backend as Backend>::DescriptorSetLayout> =
        //         vec![device.create_descriptor_set_layout(
        //             bindings,
        //             immutable_samplers,
        //         )];

        //     let push_constants =
        //         Vec::<(pso::ShaderStageFlags, std::ops::Range<u32>)>::new();

        //     let layout = device.create_pipeline_layout(
        //         &descriptor_set_layouts,
        //         push_constants,
        //     );

        //     let subpass = pass::Subpass {
        //         index: 0,
        //         main_pass: render_pass,
        //     };

        //     // let flags = pso::PipelineCreationFlags::empty();

        //     // let parent = pso::BasePipeline::None;

        //     let gfx_pipeline = {
        //         let desc = pso::GraphicsPipelineDesc::new(
        //             shaders,
        //             Primitive::TriangleList,
        //             rasterizer,
        //             &layout,
        //             subpass
        //         );


        //     // let vertex_buffers: Vec<pso::VertexBufferDesc> = Vec::new();
        //     // let attributes: Vec<pso::AttributeDesc> = Vec::new();

        //         device
        //             .create_graphics_pipeline(&desc, None)
        //             .expect("failed to create graphics pipeline!")
        //     };

        //     (descriptor_set_layouts, layout, gfx_pipeline)
        // };

        // device.destroy_shader_module(vert_shader_module);
        // device.destroy_shader_module(frag_shader_module);

        // (descriptor_set_layouts, pipeline_layout, gfx_pipeline)
        //
    }

    pub fn create_sync_objects(
        device: &<backend::Backend as Backend>::Device,
    ) -> (
        Vec<<backend::Backend as Backend>::Semaphore>,
        Vec<<backend::Backend as Backend>::Semaphore>,
        Vec<<backend::Backend as Backend>::Fence>,
    ) {
        let mut image_available_semaphores: Vec<
            <backend::Backend as Backend>::Semaphore,
        > = Vec::new();
        let mut render_finished_semaphores: Vec<
            <backend::Backend as Backend>::Semaphore,
        > = Vec::new();
        let mut in_flight_fences: Vec<
            <backend::Backend as Backend>::Fence,
        > = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores.push(device.create_semaphore());
            render_finished_semaphores.push(device.create_semaphore());
            in_flight_fences.push(device.create_fence(true));
        }

        (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        )
    }
}
