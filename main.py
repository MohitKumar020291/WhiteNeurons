import hydra
from pipeline import apply_mask
from omegaconf import DictConfig
from pipeline.run_pipeline import run
from pipeline.schema import PipelineConfig

@hydra.main(config_path="config", config_name="pipeline_config")
def main(cfg: DictConfig):
    import os
    # Changing the directory as hydra makes its own directory - output - not sure why and how it works
    os.chdir(hydra.utils.get_original_cwd())
    validated = PipelineConfig(**cfg)
    run(**validated.dict())

if __name__ == '__main__':
    main()