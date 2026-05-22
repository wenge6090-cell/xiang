/// Build script for xiang-experiments.
/// Mirrors xiang-llm's build.rs to detect llama.cpp and set
/// `cfg(feature = "llama_backend")` so experiment code can
/// conditionally compile the real-model backend path.
fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");

    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let project_root = crate_dir.parent().and_then(|p| p.parent());

    let llama_dir = std::env::var("LLAMA_CPP_DIR")
        .ok()
        .or_else(|| {
            project_root.map(|root| root.join("llama.cpp").to_string_lossy().to_string())
        });

    let Some(dir) = llama_dir else {
        return;
    };

    let p = std::path::Path::new(&dir);
    // Windows: DLL in build/bin/Release
    let lib_win = p.join("build").join("bin").join("Release").join("llama.dll");
    // Linux: lib in build/bin
    let lib_linux = p.join("build").join("bin").join("libllama.so");

    if lib_win.exists() || lib_linux.exists() {
        if lib_win.exists() {
            println!("cargo:rustc-link-search=native={}", p.join("build").join("bin").join("Release").display());
        }
        if lib_linux.exists() {
            println!("cargo:rustc-link-search=native={}", p.join("build").join("bin").display());
        }
        println!("cargo:rustc-link-search=native={}", p.join("build").display());
        println!("cargo:rustc-cfg=feature=\"llama_backend\"");
    }
}
