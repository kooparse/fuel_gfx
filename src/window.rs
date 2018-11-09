use winit::{EventsLoop, Window};

pub struct WindowState {
    pub window: Window,
    pub events_loop: Option<EventsLoop>,
}
