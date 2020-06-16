#If google colaboratory is used for the entire project, use this code after uploading the zipfile containing all relevant files onto your google drive


from zipfile import ZipFile
file_name="/content/drive/My Drive/Name-of-zip-file-you-uploaded"
with ZipFile(file_name,"r") as zip:
  zip.extractall()
  
