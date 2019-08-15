from sys import argv
from spectacle import io, analyse

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
results_readnoise = results/"readnoise"

isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso)
colours     = io.load_colour(stacks)

table = analyse.statistics(stds, prefix_column=isos, prefix_column_header="ISO")
print(table)

for ISO, std in zip(isos, stds):
    analyse.plot_histogram_RGB(std, colours, xlim=(0, 15), xlabel="Read noise (norm. ADU)", saveto=results_readnoise/f"readnoise_ADU_histogram_iso{ISO}.pdf")
    analyse.plot_gauss_maps(std, colours, colorbar_label="Read noise (norm. ADU)", saveto=results_readnoise/f"readnoise_ADU_map_iso{ISO}.pdf")
