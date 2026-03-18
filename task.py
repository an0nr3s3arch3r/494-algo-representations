import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    df = pl.read_csv("/Users/knight/uw/info494/494-algo-representations/notes-00000.tsv", separator="\t", ignore_errors=True)
    print(f"Total records: {df.height}")
    return


if __name__ == "__main__":
    app.run()
