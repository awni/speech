
all: warp transduce kenlm

.deps:
	sudo apt-get install libboost-all-dev cmake -y

warp:
	git clone https://github.com/awni/warp-ctc.git libs/warp-ctc 
	cd libs/warp-ctc; mkdir build; cd build; cmake ../ && make; \
		cd ../pytorch_binding; python build.py

# TODO, awni, put this into a package
transduce:
	git clone https://github.com/awni/transducer.git libs/transducer
	cd libs/transducer; python build.py

kenlm:
	git clone https://github.com/kpu/kenlm.git libs/kenlm
	cd libs/kenlm; mkdir build; cd build; cmake ../ && make;