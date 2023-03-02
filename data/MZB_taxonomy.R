library(taxize)

taxa_list <- c("acari", "baetidae", "blephariceridae", "chironomidae", "diptera", "ephemeroptera", 
               "heptageniidae", "isoperla", "leuctra", "oligochaeta", "protonemura", "amphinemura", 
               "baetis", "brachyptera", "coleoptera", "ephemerellidae", "hydropsychidae", "leptophlebiidae", 
               "leuctridae", "plecoptera", "simuliidae")

ncbi_id <- get_ids(taxa_list, db = "ncbi")

taxonomy <- tax_name(taxa_list, get = c("kingdom", "phylum", "class", "subclass", "order", "suborder", "family", "genus"), db = "ncbi")
taxonomy

write.csv(taxonomy, file = "../data/MZB_taxonomy.csv")
