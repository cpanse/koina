import triton_python_backend_utils as pb_utils
import triton_python_backend_utils as pb_utils
import numpy as np
import json
import re


def peak_pos_xl_cms2(unmod_seq: str, crosslinker_position: int) -> list:
    """
    Determines the positions of all potential normal and xl fragments within the vector generated by generate_annotation_matrix.

    This fuction is used only for cleavable crosslinked peptides.

    :param unmod_seq: Un modified peptide sequence
    :param crosslinker_position: The position of crosslinker
    :raises ValueError: if Peptides exceeding a length of 30.
    :return: position of diffrent fragments as list
    """
    peaks_y = []
    peaks_b = []
    peaks_yshort = []
    peaks_bshort = []
    peaks_ylong = []
    peaks_blong = []

    if len(unmod_seq) < 31:
        if crosslinker_position != 1:
            peaks_b = np.array([3, 4, 5])
            peaks_b = np.tile(peaks_b, crosslinker_position - 1) + np.repeat(
                np.arange(crosslinker_position - 1) * 6, 3
            )
            first_pos_ylong = (
                (len(unmod_seq) - crosslinker_position) * 6
            ) + 174  # fisrt  position for ylong
            peaks_ylong = np.arange(first_pos_ylong, first_pos_ylong + 3)
            peaks_ylong = np.tile(peaks_ylong, crosslinker_position - 1) + np.repeat(
                np.arange(crosslinker_position - 1) * 6, 3
            )

        if len(unmod_seq) != crosslinker_position:
            peaks_y = [0, 1, 2]
            peaks_y = np.tile(
                peaks_y, len(unmod_seq) - crosslinker_position
            ) + np.repeat(np.arange(len(unmod_seq) - crosslinker_position) * 6, 3)
            first_pos_blong = (
                ((crosslinker_position - 1) * 6) + 174 + 3
            )  # fisrt  position for blong
            peaks_blong = [first_pos_blong, first_pos_blong + 1, first_pos_blong + 2]
            peaks_blong = np.arange(first_pos_blong, first_pos_blong + 3)
            peaks_blong = list(
                np.tile(peaks_blong, len(unmod_seq) - crosslinker_position)
                + np.repeat(np.arange(len(unmod_seq) - crosslinker_position) * 6, 3)
            )

        peaks_yshort = [x - 174 for x in peaks_ylong]
        peaks_bshort = [x - 174 for x in peaks_blong]
        peaks_range = (
            list(peaks_y)
            + list(peaks_b)
            + list(peaks_yshort)
            + list(peaks_bshort)
            + list(peaks_ylong)
            + list(peaks_blong)
        )
        peaks_range.sort()
    else:
        raise ValueError(
            f"Peptides exceeding a length of 30 are not supported: {len(unmod_seq)}"
        )

    return (
        peaks_range,
        peaks_y,
        peaks_b,
        peaks_yshort,
        peaks_bshort,
        peaks_ylong,
        peaks_blong,
    )


def find_crosslinker_position(peptide_sequence: str):
    peptide_sequence = re.sub(r"\[UNIMOD:(?!1896|1884\]).*?\]", "", peptide_sequence)
    crosslinker_position = re.search(r"K(?=\[UNIMOD:(?:1896|1884)\])", peptide_sequence)
    crosslinker_position = crosslinker_position.start() + 1
    return crosslinker_position


def gen_annotation_linear_pep():
    ions = [
        "y",
        "b",
    ]
    charges = ["1", "2", "3"]
    positions = [x for x in range(1, 30)]
    annotation = []
    for pos in positions:
        for ion in ions:
            for charge in charges:
                annotation.append(ion + str(pos) + "+" + charge)
    return annotation


def gen_annotation_xl(unmod_seq: str, crosslinker_position: int):
    annotations = gen_annotation_linear_pep()
    annotation = np.concatenate((annotations, annotations))
    annotation = annotation.tolist()
    (
        peaks_range,
        peaks_y,
        peaks_b,
        peaks_yshort,
        peaks_bshort,
        peaks_ylong,
        peaks_blong,
    ) = peak_pos_xl_cms2(unmod_seq, crosslinker_position)
    for pos in peaks_yshort:
        annotation[pos] = "y_short_" + annotation[pos][1:]
    for pos in peaks_bshort:
        annotation[pos] = "b_short_" + annotation[pos][1:]
    for pos in peaks_ylong:
        annotation[pos] = "y_long_" + annotation[pos][1:]
    for pos in peaks_blong:
        annotation[pos] = "b_long_" + annotation[pos][1:]
    pos_none = (
        [num + 174 for num in peaks_y]
        + [num + 174 for num in peaks_b]
        + list(np.arange((len(unmod_seq) - 1) * 6, 174, 1))
        + list(np.arange((len(unmod_seq) - 1) * 6 + 174, 348, 1))
        + list(np.arange(2, 348, 3))
    )
    for pos in pos_none:
        annotation[pos] = "None"
    return np.array(annotation).astype(np.object_)


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(
            self.model_config, "annotation"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        logger = pb_utils.Logger
        responses = []
        annotation = np.empty((0, 348))

        for request in requests:
            batchsize = (
                pb_utils.get_input_tensor_by_name(request, "precursor_charges")
                .as_numpy()
                .shape[0]
            )

            peptide_sequences_1 = pb_utils.get_input_tensor_by_name(
                request, "peptide_sequences_1"
            ).as_numpy()

            for i in range(batchsize):
                regular_sequence = peptide_sequences_1[i][0].decode("utf-8")
                crosslinker_position = find_crosslinker_position(regular_sequence)
                unmod_seq = re.sub(r"\[.*?\]", "", regular_sequence)
                annotation_i = gen_annotation_xl(unmod_seq, crosslinker_position)
                annotation = np.vstack((annotation, annotation_i))

            t = pb_utils.Tensor("annotation", annotation)
            responses.append(pb_utils.InferenceResponse(output_tensors=[t]))

        return responses

    def finalize(self):
        pass
