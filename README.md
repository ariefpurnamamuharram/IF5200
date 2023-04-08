# Automated Chest X-Ray Report Generator in Bahasa Indonesia

This project was intended for IF5200 - Applied Research Project course project, Informatics Master Program, School of
Electrical Engineering and Informatics, Institut Teknologi Bandung.

## Before Having Fun

Please install the required packages describe in the `requirements.txt` file. The default production models located in `sys/models` are stored using Git LFS. You may need to do `git lfs pull` to pull out the models.

To avoid any local pull conflicts, it is recommended to `git update-index --assume-unchanged` to `experiments.ipynb` and `evaluation.ipynb`. This allows git not to track updates to the desired objects. To revert back, please use `git update-index --no-assume-unchanged`.

## Local Deployment

To run local deployment, please execute this command.

```
python serve.py --port 8080
```

## Contributors

1. Arief Purnama Muharram (23521013)
2. Hollyana Puteri Haryono (23522013)
3. Abassi Haji Juma (23522701)