# Automated Chest X-Ray Report Generator in Bahasa Indonesia
This project was intended for IF5200 - Applied Research Project course project, Informatics Master Program, School of Electrical Engineering and Informatics, Institut Teknologi Bandung.

## Before Having Fun
Please install the required packages describe in the `requirements.txt` file or you can do it easily by executing the `install-packages.sh` shell script.

## Example Usages
Without GPU
```
python generate_report.py \
    --image datasets/samples_cxr-images_128x128/00000001_000.png
```
with GPU
```
python generate_report.py \
    --image datasets/samples_cxr-images_128x128/00000001_000.png \
    --device cuda:0
```

## Example output
```
Using GPU!
Device name: Quadro RTX 5000

Results:
Pada foto radiologi dada yang diterima diperoleh temuan-temuan sebagai berikut: 
Bentuk jantung tampak baik, tidak ditemukan tanda-tanda kardiomegali. Tidak tampak gambaran efusi pada lapang paru. 

Done!
```

## Contributors
1. Arief Purnama Muharram
2. Hollyana Puteri Haryono
3. Abassi Haji Juma