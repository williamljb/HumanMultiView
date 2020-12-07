# ---------------------------
# ----- SET YOUR PATH!! -----
# ---------------------------
# This is the directory that contains README.txt
LSP_DIR=/scratch1/hmr_multiview/data/lsp_dataset

# This is the directory that contains README.txt
LSP_EXT_DIR=/scratch1/hmr_multiview/data/lsp_extended

# This is the directory that contains 'images' and 'annotations'
MPII_DIR=/scratch1/hmr_multiview/data/mpii

# This is where you want all of your tf_records to be saved:
DATA_DIR=/scratch1/hmr_multiview/tf_datasets/

# This is the directory that contains README.txt, S1..S8, etc
MPI_INF_3DHP_DIR=/scratch1/mpi_inf_3dhp/data/
# ---------------------------

# This is the directory that contains 0_dress/karate, etc
SYNTHETIC_DIR=/scratch1/wgan/results
# ---------------------------

MOSH_DIR=/scratch1/hmr_multiview/neutrMosh/


# ---------------------------
# Run each command below from this directory. I advice to run each one independently.
# ---------------------------
# ----- LSP -----
# python -m src_ortho.datasets.lsp_to_tfrecords --img_directory $LSP_DIR --output_directory $DATA_DIR/lsp

# ----- LSP-extended -----
# python -m src_ortho.datasets.lsp_to_tfrecords --img_directory $LSP_EXT_DIR --output_directory $DATA_DIR/lsp_ext

# ----- MPII -----
# python -m src_ortho.datasets.mpii_to_tfrecords --img_directory $MPII_DIR --output_directory $DATA_DIR/mpii

# ----- MPI-INF-3DHP -----
# python -m src_ortho.datasets.mpi_inf_3dhp_to_tfrecords --data_directory $MPI_INF_3DHP_DIR --output_directory $DATA_DIR/mpi_inf_3dhp
# python -m src_ortho.datasets.mpi_inf_3dhp_to_tfrecords --split val --data_directory $MPI_INF_3DHP_DIR --output_directory $DATA_DIR/mpi_inf_3dhp
# python -m src_ortho.datasets.mpi_inf_3dhp_test_to_tfrecords --data_directory $MPI_INF_3DHP_DIR --output_directory $DATA_DIR/mpi_inf_3dhp

# ----- synthetic -----
python -m sr_orthoc.datasets.synthetic_to_tfrecords --data_directory $SYNTHETIC_DIR --output_directory $DATA_DIR/synthetic
# python -m src_ortho.datasets.synthetic_to_tfrecords --split val --data_directory $SYNTHETIC_DIR --output_directory $DATA_DIR/synthetic

# ----- COCO -----
# python -m src_ortho.datasets.coco_to_tfrecords --data_directory /scratch1/hmr_multiview/coco/data --output_directory $DATA_DIR/coco



# ----- Mosh data, for each dataset -----
# CMU:
# python -m src_ortho.datasets.smpl_to_tfrecords --dataset_name 'neutrSMPL_CMU'

# # H3.6M:
# python -m src_ortho.datasets.smpl_to_tfrecords --dataset_name 'neutrSMPL_H3.6'

# # jointLim:
# python -m src_ortho.datasets.smpl_to_tfrecords --dataset_name 'neutrSMPL_jointLim'
