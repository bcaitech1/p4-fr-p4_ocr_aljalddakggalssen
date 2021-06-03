import editdistance
import numpy as np
import torch

def word_error_rate(predicted_outputs, ground_truths):
	sum_wer=0.0
	for output,ground_truth in zip(predicted_outputs,ground_truths):
		output=output.split(" ")
		ground_truth=ground_truth.split(" ")
		distance = editdistance.eval(output, ground_truth)
		length = max(len(output),len(ground_truth))
		sum_wer+=(distance/length)
	return sum_wer


def sentence_acc(predicted_outputs, ground_truths):
	correct_sentences=0
	for output,ground_truth in zip(predicted_outputs,ground_truths):
		if np.array_equal(output,ground_truth):
			correct_sentences+=1
	return correct_sentences

def correct_symbol(predictions, expected):
	mask = expected != -1
	correct = predictions == expected
	using = correct & mask
	ret = torch.sum(using, dim=(0, 1)).item()
	return ret

