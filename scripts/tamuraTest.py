from scripts.utils.texture.tamura import get_tamura_features, saveTamuraFeatures

if __name__ == '__main__':

    inputimagepath = r'D:\Dokumenty\CBIR\CorelDB\pl_flower\84010.jpg'
    features = get_tamura_features(inputimagepath)
    print(features)
    all_features = saveTamuraFeatures(inputimagepath)
