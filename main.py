import click
import yaml
from mergekit.config import MergeKitConfig
from mergekit.merger import MergeKitMerger
from mergekit.utils import load_models, save_model

@click.command()
@click.argument("config_file")
@click.argument("out_path")
def main(config_file: str, out_path: str):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    
    config = MergeKitConfig(**config_dict)
    
    merger = MergeKitMerger(config)
    
    models = load_models(config.models)
    
    merged_model = merger.merge(models)
    
    save_model(merged_model, out_path)

if __name__ == "__main__":
    main()