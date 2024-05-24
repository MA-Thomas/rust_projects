/TESLA_IEDB/modeling_and_aucs.py is the python script that calls rust functions.

/src/lib.rs is the main rust library. 
Auxiliary functions are in /src/lib_io.rs and /src/lib_rust_function_versions.rs 

To run the code using vscode:
1. Clone the repo form github to your local machine
2. Get started with rust in visual studio code. Follow 0:50 - 1:59 from https://www.youtube.com/watch?v=OX9HJsJUDxA&list=PLai5B987bZ9CoVR-QEIN9foz4QCJ0H2Y8
3. From the terminal, pip install maturin (this is used to publish rust binaries as python packages). See https://www.youtube.com/watch?v=DpUlfWP_gtg&t=290s for more background info.
4. Within the vscode terminal, navigate to /immunogenicity_rust and run "maturin develop" (you can also run "matruin develop --release" for more optimized binaries. There is a noticable runtime improvement.)
5. run "python TESLA_IEDB/modeling_immunogenicity.py"
