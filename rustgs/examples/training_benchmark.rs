use rustgs::gpu_available;

#[derive(Debug, Clone)]
struct BenchmarkArgs {
    width: usize,
    height: usize,
    frame_count: usize,
    gaussian_count: usize,
    warmup_steps: usize,
    measured_steps: usize,
    smoke_iterations: usize,
    litegs_mode: bool,
    json: bool,
}

impl Default for BenchmarkArgs {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            frame_count: 3,
            gaussian_count: 128,
            warmup_steps: 2,
            measured_steps: 5,
            smoke_iterations: 8,
            litegs_mode: false,
            json: false,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;

    if !gpu_available() {
        eprintln!("GPU unavailable in current environment; benchmark skipped.");
        return Ok(());
    }

    if args.json {
        println!(
            "{{\"status\":\"unavailable\",\"reason\":\"legacy metal benchmark removed during wgpu migration\"}}"
        );
    } else {
        println!("training benchmark example is not implemented for the post-migration wgpu path");
        println!(
            "requested litegs_mode={} fixture={}x{} frames={} gaussians={} warmup={} measure={} smoke_iters={}",
            args.litegs_mode,
            args.width,
            args.height,
            args.frame_count,
            args.gaussian_count,
            args.warmup_steps,
            args.measured_steps,
            args.smoke_iterations
        );
    }

    Ok(())
}

fn parse_args() -> Result<BenchmarkArgs, Box<dyn std::error::Error>> {
    let mut spec = BenchmarkArgs::default();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--width" => spec.width = parse_next(&mut args, "--width")?,
            "--height" => spec.height = parse_next(&mut args, "--height")?,
            "--frames" => spec.frame_count = parse_next(&mut args, "--frames")?,
            "--gaussians" => spec.gaussian_count = parse_next(&mut args, "--gaussians")?,
            "--warmup" => spec.warmup_steps = parse_next(&mut args, "--warmup")?,
            "--measure" => spec.measured_steps = parse_next(&mut args, "--measure")?,
            "--smoke-iters" => spec.smoke_iterations = parse_next(&mut args, "--smoke-iters")?,
            "--litegs-mode" => spec.litegs_mode = true,
            "--json" => spec.json = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unsupported argument '{other}'").into());
            }
        }
    }

    Ok(spec)
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, Box<dyn std::error::Error>>
where
    T::Err: std::fmt::Display,
{
    let value = args
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))?;
    value
        .parse::<T>()
        .map_err(|err| format!("invalid value for {flag}: {err}").into())
}

fn print_help() {
    println!("Usage: cargo run --example training_benchmark -- [options]");
    println!("  --width <n>         input width (default: 64)");
    println!("  --height <n>        input height (default: 64)");
    println!("  --frames <n>        frame count (default: 3)");
    println!("  --gaussians <n>     gaussian count (default: 128)");
    println!("  --warmup <n>        warmup steps (default: 2)");
    println!("  --measure <n>       measured steps (default: 5)");
    println!("  --smoke-iters <n>   smoke training iterations (default: 8)");
    println!("  --litegs-mode       enable LiteGS-compatible mode");
    println!("  --json              print JSON output");
}
