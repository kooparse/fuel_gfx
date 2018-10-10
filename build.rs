use glsl_to_spirv as glsl;
use std::io::Read;

use std::error::Error;

fn main() -> Result<(), Box<Error>> {
    use self::glsl::ShaderType;

    // Tell the build script to only run again if we change our source shaders
    println!("cargo:rerun-if-changed=assets/shaders");

    // Create destination path if necessary
    std::fs::create_dir_all("assets/shaders")?;

    for entry in std::fs::read_dir("assets/glsl")? {
        let entry = entry?;

        if entry.file_type()?.is_file() {
            let in_path = entry.path();

            // Support only vertex and fragment shaders currently
            let shader_type = in_path.extension().and_then(|ext| {
                match ext.to_string_lossy().as_ref() {
                    "vert" => Some(ShaderType::Vertex),
                    "frag" => Some(ShaderType::Fragment),
                    _ => None,
                }
            });

            if let Some(shader_type) = shader_type {
                let source = std::fs::read_to_string(&in_path)?;
                let mut compiled_file =
                    self::glsl::compile(&source, shader_type)?;

                let mut compiled_bytes = Vec::new();
                compiled_file.read_to_end(&mut compiled_bytes)?;

                let out_path = format!(
                    "assets/shaders/{}.spv",
                    in_path.file_name().unwrap().to_string_lossy()
                );

                std::fs::write(&out_path, &compiled_bytes)?;
            }
        }
    }

    Ok(())
}
