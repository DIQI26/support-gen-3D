*Please note that this repository was copied from an educational account and is used to showcase the project results to third parties.*

**Support Generator for 3D Printing**

Description

This project is a tool that generates support structures for 3D printing. It currently supports two file formats: STL and SLC. Depending on the file type provided by the user, it uses different algorithms to process the input.

Features

Processes STL and SLC file types.
Uses the corresponding algorithm based on file type:
stl_main() for STL files.
slc_main() for SLC files.

Install dependencies using Poetry:
``` 
poetry install 
```

To run the program using Poetry:

Use the following command to start the combined script:

``` 
poetry run combined 
```

The script will prompt you to enter a file path. You need to provide a valid STL or SLC file. 

Based on the file type:

If you provide an STL file, the stl_main() function will process it.
If you provide an SLC file, the slc_main() function will process it.
The program will continue to run until you input a valid file.

To run tests:

```
poetry run pytest 
```
