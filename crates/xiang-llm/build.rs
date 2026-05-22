/// Build script for xiang-llm.
///
/// Detects llama.cpp installation and sets the `llama` cfg flag
/// to conditionally enable the real llama backend.
///
/// Supported discovery paths (in order):
///   1. Environment variable `LLAMA_CPP_DIR`
///   2. Default path: ../llama.cpp (relative to this crate)

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");

    // Determine project root relative to this build.rs location
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let project_root = crate_dir.parent().and_then(|p| p.parent());

    let llama_dir = std::env::var("LLAMA_CPP_DIR").ok()
        .or_else(|| {
            project_root.map(|root| root.join("llama.cpp").to_string_lossy().to_string())
        });

    let found = if let Some(ref dir) = llama_dir {
        check_llama_in_dir(dir)
    } else {
        false
    };

    if found {
        if let Some(ref dir) = llama_dir {
            let p = std::path::Path::new(dir);
            // Windows: DLL in build/bin/Release, .lib in build/src/Release
            let release_bin = p.join("build").join("bin").join("Release");
            let release_src = p.join("build").join("src").join("Release");
            if release_bin.exists() {
                println!("cargo:rustc-link-search=native={}", release_bin.display());
            }
            if release_src.exists() {
                println!("cargo:rustc-link-search=native={}", release_src.display());
            }
            // Linux: lib in build/bin
            let build_bin = p.join("build").join("bin");
            if build_bin.exists() {
                println!("cargo:rustc-link-search=native={}", build_bin.display());
            }
        }
        println!("cargo:rustc-link-lib=llama");
        println!("cargo:rustc-cfg=feature=\"llama\"");
        println!("cargo:rustc-cfg=feature=\"llama_backend\"");
    }

    // Link to the appropriate C++ runtime based on platform.
    // Windows/MSVC uses its own C++ runtime (msvcprt), no extra link needed.
    // Linux/GCC needs explicit -lstdc++.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn check_llama_in_dir(dir: &str) -> bool {
    let p = std::path::Path::new(dir);
    // Check either source headers or built library
    let header = p.join("include").join("llama.h");
    // Windows: llama.dll in build/bin/Release
    let lib_win = p.join("build").join("bin").join("Release").join("llama.dll");
    // Linux: libllama.so in build/bin
    let lib_linux = p.join("build").join("bin").join("libllama.so");
    (header.exists() && (lib_win.exists() || lib_linux.exists()))
        || lib_win.exists() || lib_linux.exists()
}
