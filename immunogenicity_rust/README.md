/TESLA_IEDB/modeling_and_aucs.py is the python script that calls rust functions.

/src/lib.rs is the main rust library. 
Auxiliary functions are in /src/lib_io.rs, /src/lib_rust_function_versions.rs, etc. 

Notes:
Distances calculations to self epitope set takes <20s per query epitope.
Computing the immunogenicity factors (including the distance calculations) at ~400 parameter sets takes ~3min per query epitope.


To run the code using vscode:

1. Clone the repo form github to your local machine
2. Get started with rust in visual studio code. Follow 0:50 - 1:59 from https://www.youtube.com/watch?v=OX9HJsJUDxA&list=PLai5B987bZ9CoVR-QEIN9foz4QCJ0H2Y8
3. From the terminal, pip install maturin (this is used to publish rust binaries as python packages). See https://www.youtube.com/watch?v=DpUlfWP_gtg&t=290s for more background info.
4. Within the vscode terminal, navigate to /immunogenicity_rust and run "maturin develop" (you can also run "matruin develop --release" for more optimized binaries. There is a noticable runtime improvement.)
5. Ensure the input files to TESLA_IEDB/modeling_immunogenicity.py are available on your computer, update the default paths used in the argparse section.
6. run "python TESLA_IEDB/modeling_immunogenicity.py"
