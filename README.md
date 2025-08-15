# BOL-software-test-
using open AI in the background to manually read bill of lading and convert to json file
This is a small program that takes scanned Bill of Lading (BOL) PDFs or images, 
reads the text using OCR, and pulls out the important details like BOL number, 
dates, weights, and carrier info. It then saves this data into:

- A JSON file (for computers to use)

You can run it locally or in Docker, drop in a scanned BOL, and see the extracted data right away. 
Later, this same setup can plug into RPA software without big changes.
