mod constants;
mod gfx;
mod vertex;
mod window;

use crate::constants::{APP_NAME, WINDOW_HEIGHT, WINDOW_WIDTH, MAX_FRAMES_IN_FLIGHT};
use crate::gfx::HalState;
use crate::window::WindowState;
use winit::{
    Event, EventsLoop, VirtualKeyCode, Window, WindowBuilder, WindowEvent,
};

struct Program {
    hal_state: gfx::HalState,
    window_state: WindowState,
}

impl Program {
    pub fn new() -> Program {
        // Initialize our window stuff.
        let window_state = Program::init_window();
        let hal_state = Program::init_hal(&window_state.window);

        Program {
            window_state,
            hal_state,
        }
    }

    fn init_window() -> WindowState {
        let events_loop = EventsLoop::new();
        let window = WindowBuilder::new()
            .with_title(APP_NAME)
            .with_dimensions((WINDOW_WIDTH, WINDOW_HEIGHT).into())
            .build(&events_loop)
            .expect("Failed to create window.");

        WindowState {
            events_loop: Some(events_loop),
            window,
        }
    }

    fn init_hal(window: &Window) -> HalState {
        let instance = HalState::create_instance();
        let mut adapter = HalState::pick_adapter(&instance);
        let mut surface = HalState::create_surface(&instance, window);
        let (device, queue_group, command_queues) =
            HalState::create_device_with_graphics_queues(
                &mut adapter,
                &surface,
            );
        
        let (swapchain, extent, backbuffer, format) =
            HalState::create_swapchain(&adapter, &device, &mut surface, None);
        let frame_images =
            HalState::create_image_views(backbuffer, format, &device);
        let render_pass = HalState::create_render_pass(&device, Some(format));


        let (descriptor_set_layouts, pipeline_layout, gfx_pipeline, vertex_buffer, _buffer_memory) =
            HalState::create_graphics_pipeline(&device, &adapter, extent, &render_pass);

        let swapchain_framebuffers = HalState::create_framebuffers(
            &device,
            &render_pass,
            &frame_images,
            extent,
        );
        let mut command_pool =
            HalState::create_command_pool(&device, queue_group.family());

        let submission_command_buffers = HalState::create_command_buffers(
            &mut command_pool,
            &render_pass,
            &swapchain_framebuffers,
            extent,
            &gfx_pipeline,
            vertex_buffer
        );
        let (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        ) = HalState::create_sync_objects(&device);

        HalState {
            in_flight_fences,
            render_finished_semaphores,
            image_available_semaphores,
            submission_command_buffers,
            command_pool,
            swapchain_framebuffers,
            gfx_pipeline,
            descriptor_set_layouts,
            pipeline_layout,
            render_pass,
            frame_images,
            _format: format,
            swapchain,
            command_queues,
            device,
            _surface: surface,
            _adapter: adapter,
            _instance: instance,
        }
    }

    pub fn main_loop(&mut self) {
        let mut current_frame: usize = 0;
        let mut quitting = false;

        let mut events_loop = self
            .window_state
            .events_loop
            .take()
            .expect("No event loop found");

        while !quitting {
            events_loop.poll_events(|event| {
                match event {
                    // handling keyboard event
                    Event::WindowEvent { event, .. } => match event {
                        WindowEvent::KeyboardInput { input, .. } => {
                            if let Some(VirtualKeyCode::Escape) =
                                input.virtual_keycode
                            {
                                quitting = true;
                            }
                        }
                        WindowEvent::CloseRequested => {
                            quitting = true;
                        }
                        _ => (),
                    },
                    _ => (),
                }

                self.hal_state.draw(current_frame);
                current_frame = ( current_frame + 1 ) % MAX_FRAMES_IN_FLIGHT;

                if quitting {
                    println!("Quitting...")
                }
            })
        }
    }

    pub fn cleanup(self) {
        self.hal_state.cleanup();
    }
}

fn main() {
    let mut prog = Program::new();
    // Start the main loop.
    prog.main_loop();
    prog.cleanup();
}
