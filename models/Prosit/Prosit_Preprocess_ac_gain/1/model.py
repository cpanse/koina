import triton_python_backend_utils as pb_utils
import numpy as np
from modifications import ProformaParser, Unimod
import re
import json

dict_index_pos = {"H": 0 ,"C": 1, "N": 2, "O": 3,"P": 4, "S": 5 }

unimod = Unimod()


def atom_count_str_list(atom_count, atom_count_list):
    atom_count = atom_count[1:-1]
    atom_count = atom_count.split(" ")
    for atoms in atom_count:
        count = re.findall(r"\d+", atoms)
        if len(count) > 0:
            atom_count_list[dict_index_pos[atoms[0]]] += int(count[0])
        else:
            atom_count_list[dict_index_pos[atoms[0]]] += 1
    return atom_count_list


def get_ac(seq):
    seq = unimod.lookup_sequence_m(
        ProformaParser.parse_sequence(seq), keys_to_lookup=["delta_composition"]
    )
    aa_ac_placeholder = np.zeros([32, 6])
    aa_ac_list = []
    for aa in seq:
        current_ac = [0, 0, 0, 0, 0, 0]
        if aa[1] != "-":
            current_ac = atom_count_str_list(aa[1], current_ac)
        aa_ac_list.append(current_ac)
    aa_ac_placeholder[: len(aa_ac_list),] = aa_ac_list
    return aa_ac_placeholder


def get_ac_all(sequences):
    aa_ac = [get_ac(seq) for seq in sequences]
    return aa_ac


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "ac_gain"
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
            t = pb_utils.Tensor("ac_gain", fill.astype(self.output_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))
        return responses

    def finalize(self):
        pass
