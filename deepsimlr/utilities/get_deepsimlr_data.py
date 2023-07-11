import torchvision
import os

def get_deepsimlr_data(file_id=None,
                       target_file_name=None,
                       deepsimlr_cache_directory=None):

    """
    Download data such as prefabricated templates and spatial priors.

    Arguments
    ---------

    file_id string
        One of the permitted file ids or pass "show" to list all
        valid possibilities. Note that most require internet access
        to download.

    target_file_name string
       Optional target filename.

    deepsimlr_cache_directory string
       Optional target output.  If not specified these data will be downloaded
       to the subdirectory ~/.deepsimlr/.

    Returns
    -------
    A filename string

    Example
    -------
    >>> template_file = get_deepsimlr_data('biobank')
    """

    def switch_data(argument):
        switcher = {
            "biobank": "https://ndownloader.figshare.com/files/22429242",
            "croppedMni152": "https://ndownloader.figshare.com/files/22933754"
        }
        return(switcher.get(argument, "Invalid argument."))

    if file_id == None:
        raise ValueError("Missing file id.")

    valid_list = ("biobank",
                  "croppedMni152",
                  "show")

    if not file_id in valid_list:
        raise ValueError("No data with the id you passed - try \"show\" to get list of valid ids.")

    if file_id == "show":
       return(valid_list)

    url = switch_data(file_id)

    if target_file_name == None:
        target_file_name = file_id + ".nii.gz"

    if deepsimlr_cache_directory is None:
        deepsimlr_cache_directory = os.path.join(os.path.expanduser('~'), '.deepsimlr/')
    target_file_name_path = deepsimlr_cache_directory + target_file_name

    if not os.path.exists(target_file_name_path):
        torchvision.datasets.utils.download_url(url,
                                                deepsimlr_cache_directory,
                                                target_file_name)

    return(target_file_name_path)
