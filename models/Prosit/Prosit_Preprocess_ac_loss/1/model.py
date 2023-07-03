import triton_python_backend_utils as pb_utils
import numpy as np
from modifications import ProformaParser, Unimod
import re
import json


def get_ac(seq):
    aa_ac_placeholder = np.zeros([32, 6])
    return aa_ac_placeholder


def get_ac_all(sequences):
    aa_ac = [get_ac(seq) for seq in sequences]
    return aa_ac


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "ac_loss"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        peptide_in_str = []
        responses = []
        for request in requests:
            peptide_in = pb_utils.get_input_tensor_by_name(request, "peptide_sequences")
            peptides_ = peptide_in.as_numpy().tolist()
            peptide_in_list = [x[0].decode("utf-8") for x in peptides_]

            fill = np.array(get_ac_all(peptide_in_list))
            t = pb_utils.Tensor("ac_loss", fill.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
