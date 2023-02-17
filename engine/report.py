def make_report(cardiomegaly: int = 0, effusion: int = 0) -> str:
    
    try:
        report_text = f'Pada foto radiologi dada yang diterima diperoleh temuan-temuan sebagai berikut: {"Bentuk jantung tampak baik, tidak ditemukan tanda-tanda kardiomegali" if bool(cardiomegaly) else "Terdapat gambaran kardiomegali, CTR > 50%"}. {"Tidak tampak gambaran efusi" if bool(effusion) else "Tampak gambaran efusi"} pada lapang paru.'
    except:
        raise ValueError('Report error!')
    
    return report_text