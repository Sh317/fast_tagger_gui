import sys
father_path = "C:/Users/MITBeamBox_01/Desktop/SOFTWARE/Tagger/TW_DAQ"
sys.path.append(father_path)
from fast_tagger_gui.src.tag_interface import Tagger

def test_tagger():
    tagger = Tagger()
    tagger.set_trigger_falling()
    tagger.set_trigger_level(-0.5)
    tagger.start_reading()
    n_measurements = int(1e3)
    while n_measurements<1e3:
        data = tagger.read()
        print(data)
        n_measurements += 1
    tagger.stop_reading()
    
    
if __name__ == '__main__':
    test_tagger()