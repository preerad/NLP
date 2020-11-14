import zipfile
import pandas as pd


local_zip = 'C:/Users/preer/Desktop/hin-eng.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:/Users/preer/Desktop/hin-eng')
zip_ref.close()






