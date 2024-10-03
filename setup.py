
class exp_info:
    """
    Class containing the experiment information.
    """

    def __init__(self):
    # Define ctf data path and files path
        self.ph_labels = ['CH', 'NY', 'R', 'a', 'b', 'd', 'e', 'f', 'g', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's',
                          't', 'u', 'x', 'y']

        self.ph_labels_man = ['(d)o', 'A', 'AH', 'CH', 'F', 'NY', 'R', 'Y', 'a', 'ap', 'b', 'br', 'c', 'chas', 'd',
                              'de', 'e', 'es', 'f', 'g', 'h', 'i', 'k', 'l', 'lg', 'm', 'n', 'ns', 'o', 'p', 'r', 's',
                              'si', 't', 'u', 'v', 'x', 'y']
        self.ph_labels_phonet = ['B', 'D', 'F', 'G', 'N', 'T', 'a', 'b', 'd', 'e', 'f', 'i', 'j', 'jj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 'rr', 's', 't', 'tS', 'u', 'w', 'x', 'z']
        
        self.phonological_labels={
            "vocalic" : ["a","e","i","o","u", "w", "j"],
            "consonantal" : ["b", "B","d", "D","f", "F","k","l","m","n", "N","p","r","rr","s", "Z", "T","t","g", "G","tS","S","x", "jj", "J", "L", "z"],
            "back"        : ["a","o","u", "w"],
            "anterior"    : ["e","i","j"],
            "open"        : ["a","e","o"],
            "close"       : ["j","i","u", "w"],
            "nasal"       : ["m","n", "N"],
            "stop"        : ["p","b", "B","t","k","g", "G","tS","d", "D"],
            "continuant"  : ["f", "F","b", "B","tS","d", "D","s", "Z", "T","x", "jj", "J","g", "G","S","L","x", "jj", "J", "z"],
            "lateral"     :["l"],
            "flap"        :["r"],
            "trill"       :["rr"],
            "voice"       :["a","e","i","o","u", "w","b", "B","d", "D","l","m","n", "N","rr","g", "G","L", "j"],
            "strident"    :["tS","f", "F","s", "Z", "T", "z",  "S"],
            "labial"      :["m","p","b", "B","f", "F"],
            "dental"      :["t","d", "D"],
            "velar"       :["k","g", "G"],
            "pause"       :  ["sil", "<p:>"]
            }
        
lista_aux= ["a","e","i","o","u", "w", "j","b", "B","d", "D","f", "F","k","l","m","n", "N","p","r","rr","s", "Z", "T","t","g", "G","tS","S","x", "jj", "J", "L", "z","a","o","u", "w","e","i","j","a","e","o","j","i","u", "w","m","n", "N","p","b", "B","t","k","g", "G","tS","d", "D","f", "F","b", "B","tS","d", "D","s", "Z", "T","x", "jj", "J","g", "G","S","L","x", "jj", "J", "z","l","r","rr","a","e","i","o","u", "w","b", "B","d", "D","l","m","n", "N","rr","g", "G","L", "j","tS","f", "F","s", "Z", "T", "z",  "S","m","p","b", "B","f", "F","t","d", "D","k","g", "G",  "sil", "<p:>"]
np.unique(lista_aux)

['B', 'D', 'F', 'G', 'N', 'T', 'a', 'b', 'd', 'e', 'f', 'i', 'j', 'jj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 'rr', 's', 't', 'tS', 'u', 'w', 'x', 'z']