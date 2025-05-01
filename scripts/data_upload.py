from pathlib import Path
import wandb

run = wandb.init(project="Orange Detect")

artifact_filepath = Path(r"C:\Users\Lucas\Documents\GitHub\OrangeDetect\data\processed")
  
logged_artifact = run.log_artifact(
  artifact_filepath,
  "artifact-name",
  type="dataset"
)
run.link_artifact(   
  artifact=logged_artifact,  
  target_path="wandb-registry-dataset/Images from OrangeDetect"
)
run.finish()