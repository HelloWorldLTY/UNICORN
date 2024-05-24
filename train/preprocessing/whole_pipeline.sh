export PYTHONPATH="${PYTHONPATH}:../../"
bed_path="/gpfs/radev/project/ying_rex/tl688/seq2cells/tests/individual_tsv/GTEx_v9_snRNAseq_data_GTEX-1I1GU.bed"
genome_path="../../individual_fasta/individual_genome_GTEX-1I1GU-0826-SM-GQZA2.fa"
out_tsv="./query_GTEx_v9_snRNAseq_data_GTEX-1I1GU.tsv"
out_name="enformer_out_GTEx_v9_snRNAseq_data_GTEX-1I1GU"

python create_seq_window_queries.py \
    --in $bed_path \
    --ref_genome $genome_path \
    --out $out_tsv \
    --chromosome_col 1\
    --position_col 3\
    --position_base 1 \
    --strand_col 6 \
    --group_id_col 7 \
    --additional_id_col 8 \
    --no-stitch
    
python calc_embeddings_and_targets.py \
--in_query $out_tsv \
--ref_genome $genome_path  \
--out_name $out_name \
--position_base 1 \
--add_bins 0 \
--store_text \
--store_h5 \
--targets '4675:5312'  # for all Enformer cage-seq targets