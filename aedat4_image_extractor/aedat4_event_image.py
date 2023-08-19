# You can't run below cripts directly.
# You should fit your own file route.
# But codes are not that complicated, so maybe you cna understand easily.

import numpy as np
import aedat
import cv2
import os


class Aedat4:
    def __init__(self, aedat4_path, img_save_path):
        self.decoder = aedat.Decoder(aedat4_path)
        self.img_save_path = img_save_path
        os.makedirs(self.img_save_path, exist_ok=True)

    def show_packet_nums(self):
        event_nums = 0
        for packet in self.decoder:
            if packet["stream_id"] == 0:
                event_nums += 1
            print(event_nums)

    def show_packet(self):
        for packet in self.decoder:
            print(packet)

    def extract_imgs(self):
        events = []
        index = 1
        ### img 추출 ###
        for packet in self.decoder:
            if "frame" in packet:
                image = packet["frame"]["pixels"]
                if packet["frame"]["format"] == "RGB":
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif packet["frame"]["format"] == "RGBA":
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(f"{self.img_save_path}\\{index}.png", image)
                index += 1

    def extract_events(self):
        """
            imwrite event images per event packets
        """
        event_packet_nums = 0
        for packet in self.decoder:
            if packet["stream_id"] == 0:
                event_packet_nums += 1
                p = []
                t = []
                x = []
                y = []
                for event in packet["events"]: # event = (t,x,y,p)
                    t.append(event[0])
                    x.append(event[1])
                    y.append(event[2])
                    p.append(event[3])
                    # An Event Packet
                    # Now let's draw t,x,y,p lists into a map
                maps = np.zeros((260, 346, 3), dtype=np.uint8)
                for k in range(len(p)): # No matter to use len(t/x/y) instead of len(p)
                    cx = x[k]  # Coordinate X
                    cy = y[k]  # Coordinate Y
                    if p[k]:  #  When Polarity is True (Positive):
                        maps[cy][cx] = (0, 0, 255) # Green
                    elif not p[k]: # False (Negative)
                        maps[cy][cx] = (0, 255, 0) # Red
                    # Now let's imwrite the map
                cv2.imwrite("%s\\%05d.jpg" % (self.img_save_path, event_packet_nums), maps)
                if event_packet_nums % 100 == 0:
                    print("%sth imgs created." % event_packet_nums)
        print("Done for extracting events and convert to images. %s imgs created." % event_packet_nums)

    def extract_events_frame(self, fps):
        """
            Imwrite event images according to fps
        """
        image_time_interval = (1/fps) * 1000000
        event_packet_nums = 0

        total_events_p = []
        total_events_t = []
        total_events_x = []
        total_events_y = []
        for packet in self.decoder:
            if packet["stream_id"] == 0:
                event_packet_nums += 1
                for event in packet["events"]: # event = (t,x,y,p)
                    total_events_t.append(event[0])
                    total_events_x.append(event[1])
                    total_events_y.append(event[2])
                    total_events_p.append(event[3])
        if not total_events_t:
            return
        total_timestamp_length = total_events_t[-1] - total_events_t[0]
        if total_timestamp_length <= 0 :
            print("Error, event interval cannot under than 0")
            return

        print("All packets loaded. Total Timestamp Length : %s, image_time_interval : %s" %
              (total_timestamp_length, image_time_interval))

        # Packe Loaded. 
        # Now load all events per save_interval
        img_nums = 0
        start_time = total_events_t[0]
        end_time = start_time + image_time_interval
        p = []
        x = []
        y = []
        for idx in range(len(total_events_t)):
            if start_time <= total_events_t[idx] <= end_time:
                x.append(total_events_x[idx])
                y.append(total_events_y[idx])
                p.append(total_events_p[idx])
            else:
                maps = np.zeros((260, 346, 3), dtype=np.uint8)
                for k in range(len(x)):
                    if p[k]:
                        maps[y[k]][x[k]] = (0, 0, 255)
                    elif not p[k]:
                        maps[y[k]][x[k]] = (0, 255, 0)
                cv2.imwrite("%s\\%05d.jpg" % (self.img_save_path, img_nums), maps)
                # print("%05d.jpg created." % img_nums)
                img_nums += 1
                temp = end_time
                end_time = end_time + image_time_interval
                start_time = temp
                p = []
                x = []
                y = []
        print("Done for extracting events and convert to images.")


if __name__ == "__main__":
    
    ### At here, you should fit your own route 
    ### Here is the example for aedat4 files for ImageNet-VID Datasets.
    base_path = "B:\\pkd\\Event Data\\[AEDAT] ImageNet-VID-2015\\snippets\\ILSVRC2015_VID_train_0000\\"
    aedat4_folder_path = base_path + "aedat4"
    img_base_path = base_path + "images"
    os.makedirs(img_base_path, exist_ok=True)
    for aedat4_file in os.listdir(aedat4_folder_path):
        if aedat4_file[:9] == "converted":
            continue
        if aedat4_file[-3:] == "txt":
            continue
        aedat4_file_path = aedat4_folder_path + "\\" + aedat4_file

        label_name = aedat4_file[:-7]

        event_img_save_path = img_base_path + "\\events\\" + label_name
        print(event_img_save_path)
        os.makedirs(event_img_save_path, exist_ok=True)
        video_file_path = base_path + label_name + ".mp4"

        video = cv2.VideoCapture(video_file_path)
        fps = video.get(cv2.CAP_PROP_FPS)

        aedat4 = Aedat4(aedat4_path=aedat4_file_path, img_save_path=event_img_save_path)
        aedat4.extract_events_frame(int(fps))
    #### Example End ### 

    ### You can use below codes for run this codes.
    # aedat4 = Aedat4(aedat4_path=aedat4_file_path, img_save_path=event_img_save_path)
    # aedat4.extract_events()
    # aedat4.extract_events_frame(int(fps))
    # aedat4.show_packet_nums()