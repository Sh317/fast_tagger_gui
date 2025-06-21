import threading
import time 
import epics
pv_names = ["LaserLab:spectrum_peak"]

global wavenumbers_pvs

class SpectrometreReader(threading.Thread):
    def __init__(self, refresh_rate=0.2):
        super().__init__()
        self.refresh_rate = refresh_rate
        self.spectrum = None
        self.pv_name = pv_names[0]  
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            self.spectrum = self.get_spec()
            time.sleep(self.refresh_rate)
    
    def stop(self):
        self.stop_event.set()
    
    def get_spec(self):
        try:
            spec = epics.caget(self.pv_name)
            spec = float(spec) if spec is not None else 0.00
            return spec
        except Exception as e:
            print(f"Error getting spectrum: {e}")
            print("Spectrum Disconnected!:", spec)
            
            return 0.00
        
if __name__ == "__main__":
    reader = SpectrometreReader()
    reader.start()
    time.sleep(0.5)
    reader.stop()
    print("Spectrum:", reader.spectrum)