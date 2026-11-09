import sys

import click

from any2markdown.config import load_config
from any2markdown.converter import Any2Markdown


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path(), required=False, default=None)
@click.option("-o", "--output", type=click.Path(), default=None, help="Output file path (default: stdout)")
@click.option("-c", "--config", "config_path", type=click.Path(), default=None, help="Config file path (default: $ANY2MD_CONFIG, ./any2md.yaml, repo any2md.yaml or ~/.any2md.yaml)")
@click.option("--api-key", default=None, help="API key")
@click.option("--base-url", default=None, help="Model API base URL")
@click.option("--model", default=None, help="Model name")
@click.option("--prompt", default=None, help="Model prompt")
@click.option("--temperature", default=None, type=float, help="Model temperature")
@click.option("--dpi", default=None, type=int, help="Render DPI for PDF/Office files")
@click.option("--concurrency", default=None, type=int, help="Max concurrent model calls")
@click.option("--timeout", default=None, type=int, help="Timeout per page (seconds)")
@click.option("--max-retries", default=None, type=int, help="Max retries per page")
@click.option("--table-format", "table_format", type=click.Choice(["html", "otsl"]), default=None, help="Table output format: html (convert OTSL→HTML, default) or otsl (keep).")
@click.option(
    "--parse-mode",
    type=click.Choice(["auto", "ocr", "direct"]),
    default=None,
    help=(
        "auto: parse text-based files directly and OCR the rest (default); "
        "ocr: always render and OCR; direct: require a direct parse."
    ),
)
@click.option("--debug-dir", type=click.Path(), default=None, help="Save images, prompts and responses for debugging")
def main(
    input_file,
    output_file,
    output,
    config_path,
    api_key,
    base_url,
    model,
    prompt,
    temperature,
    dpi,
    concurrency,
    timeout,
    max_retries,
    table_format,
    parse_mode,
    debug_dir,
):
    """Convert documents to Markdown using vision model.

    \b
    Usage:
      any2md input.pdf output.md
      any2md input.pdf -o output.md
      any2md input.pdf                  # prints to stdout
    """
    output_path = output or output_file

    try:
        cfg = load_config(config_path)
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    converter = Any2Markdown(
        api_key=api_key or cfg.model.api_key,
        base_url=base_url or cfg.model.base_url,
        model=model or cfg.model.name,
        prompt=prompt or cfg.model.prompt,
        temperature=temperature if temperature is not None else cfg.model.temperature,
        timeout=timeout or cfg.model.timeout,
        dpi=dpi or cfg.render.dpi,
        concurrency=concurrency or cfg.pipeline.concurrency,
        max_retries=max_retries or cfg.pipeline.max_retries,
        table_format=table_format or cfg.pipeline.table_format,
        parse_mode=parse_mode or cfg.parse.mode,
        debug_dir=debug_dir,
    )

    try:
        result = converter.convert(input_file)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if result.failed_pages:
        click.echo(
            f"Warning: {len(result.failed_pages)} page(s) failed: {result.failed_pages}",
            err=True,
        )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.markdown)
        click.echo(
            f"Written to {output_path} "
            f"({result.total_pages} pages, {result.parse_mode_used})",
            err=True,
        )
    else:
        click.echo(result.markdown)


if __name__ == "__main__":
    main()
