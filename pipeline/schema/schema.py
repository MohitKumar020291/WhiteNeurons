from pydantic import BaseModel
from typing import Union, List

# There could be multiple such classes
# How to make these error messages sound more good
class PipelineConfig(BaseModel):
    input_file: str
    output_folder: str = './ss'
    color: List[int]
    step: Union[int, float]
    show_by: str
    use_binary_mask: bool
    take_ss: bool
    ...