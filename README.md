# I Bet You Are Wrong: Gambling Adversarial Networks for Structured Semantic Segmentation

This repository contains the implementation and additional resources for the research paper "I Bet You Are Wrong: Gambling Adversarial Networks for Structured Semantic Segmentation," presented at the IEEE/CVF International Conference on Computer Vision (ICCV) in 2019 by Laurens Samson, Nanne van Noord, Olaf Booij, Michael Hofmann, Efstratios Gavves, and Mohsen Ghafoorian.

## Abstract

In our work, we address the limitations of adversarial training in structured semantic segmentation. We identify the challenge of value-based discrimination between network predictions and ground-truth annotations, which can negatively impact the learning of structural qualities and the expression of uncertainties in the network. Our approach reimagines adversarial training for semantic segmentation by shifting from a fake/real discrimination framework to a correct/incorrect training objective. We introduce a "gambler" network that learns to allocate its resources to areas where predictions are evidently incorrect, challenging the segmentation network to improve its performance without leaving clear indications of error. Our empirical evaluations on two road-scene semantic segmentation tasks demonstrate that this method enhances the network's ability to express uncertainties and improves both pixel-wise and structure-based metrics.

## Paper

The full paper can be accessed here: [ICCV 2019 Paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVRSUAD/Samson_I_Bet_You_Are_Wrong_Gambling_Adversarial_Networks_for_Structured_ICCVW_2019_paper.pdf)

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{Samson2019IBetYouAreWrong,
  title={I Bet You Are Wrong: Gambling Adversarial Networks for Structured Semantic Segmentation},
  author={Laurens Samson and Nanne van Noord and Olaf Booij and Michael Hofmann and Efstratios Gavves and Mohsen Ghafoorian},
  booktitle={The IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019}
}



