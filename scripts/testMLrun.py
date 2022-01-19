from main_ML import run_test_ML, run_test_ML_Eukliedian, run_test_ML_Maksimum, run_test_ML_Manhattan, run_test_ML_Minkowski
from test_conf import selectRandomInputImages


directory = 'D:\Dokumenty\CBIR\CorelDBML'
numberofresults = 20
numberofinputsinoneclass = 10

inputimages = selectRandomInputImages(numberofinputsinoneclass)
print('ok')
with open('Inputimsgeslist.txt', 'w') as f:
    f.write(str(inputimages))
resultE = run_test_ML_Eukliedian(numberofresults, inputimages)
print("Results Eukliedian:")
print(resultE)
resultMaks = run_test_ML_Maksimum(numberofresults, inputimages)
print("Results Maksimum:")
print(resultMaks)
resultMan = run_test_ML_Manhattan(numberofresults, inputimages)
print("Results Manhattan:")
print(resultMan)
resultMin = run_test_ML_Minkowski(numberofresults, inputimages)
print("Results Minkowski:")
print(resultMin)
