import ast 
import os
from os.path import dirname, join, exists
import yaml
from glob import glob


def load_losses(load_path):

    with open(load_path  + "losses1.txt", "r") as file:
        losses = file.read()

    with open(load_path + "val_losses1.txt", "r") as file:
        val_losses = file.read()

    losses = ast.literal_eval(losses) 
    val_losses = ast.literal_eval(val_losses)
    return losses, val_losses

class MetaManager(object):
    """
    Object to process metadata of the run
    """
    def __init__(self, run_path, restart=False, name=None, metadata=None):
        """
        Parameters:
            run_path = "Donde se va a guardar el run"
            
        """ 
        self.run_path = run_path
        self.metadata = self.create_or_load_run(run_path, restart=restart, name=name, metadata=metadata)
        if len(glob(join(self.step_path, "*"))) == 0:
            self.update_step()
        self.step_path = join(run_path, f"step{self.metadata[1]}")

    def create_run(self, run_path, restart=False, name=None, metadata = None):
        """
        Createes the directory, 
        initializes the metadata and creates the first step folder
        """
        if name is None:
            name = dirname(run_path).split(os.sep)[-1]
        if metadata is None:
            metadata = {}
            
        #if the directory exists then don't create a new run
        if not exists(run_path) or restart:
            print("Creating run")
            os.makedirs(run_path, exist_ok=True)
            meta_path = join(run_path, "metadata.yaml")
            run_metadata = { **{"name":name, "step":1}, **metadata}

            os.makedirs(join(run_path, "step1"))

            with open(meta_path, "w") as f:
                yaml.dump(run_metadata, f, default_flow_style=False)
            return run_metadata
        else: 
            print("Run already exists")

    def load_run(self, run_path):
        """
        Loads the metadata for the run
        """
        print("Loading run")
        meta_path = join(run_path, "metadata.yaml")
        with open(meta_path, "r") as f: 
            metadata = yaml.load(f)
        return metadata

    def create_or_load_run(self, run_path, restart=False, name=None, metadata=None):
        """
        If run exists loads it, if not, create one and load it
        """
        if not exists(run_path) or restart:
            run_metadata = self.create_run(run_path, restart=restart, name=name, metadata=metadata)
        else:
            run_metadata = self.load_run(run_path)
        return run_metadata

    def write_meta(self):
        meta_path = join(self.run_path, "metadata.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(self.metadata, f, default_flow_style=False)
    
    def update_step(self):
        self.metadata["step"] += 1
        self.write_meta()
        os.makedirs(join(self.run_path, f"step{self.metadata['step']}"))