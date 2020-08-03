from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()
arguments = {"keywords":"crumpled duvet bed",
"print_urls":False,
"output_directory":"data//google_images",
"chromedriver":r"C:\Users\samme\Google_Drive\Code_library\MakeYourBed\chromedriver_win32\chromedriver.exe",
"f":"jpg",
"limit":"1200",


} 
absolute_image_paths = response.download(arguments)