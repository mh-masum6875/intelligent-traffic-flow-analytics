import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["TORCH_ALLOW_UNSAFE_LOADING"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import shutil
from pathlib import Path
from datetime import datetime
start_time = datetime.now()
import cv2
import torch
import torch.backends.cudnn as cudnn
import openpyxl

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def is_daytime(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = gray.mean()
    is_day = avg_brightness > 75
    return is_day

ids = []
sort_data = {}
count_AB = {}
count_BA = {}
count_AC = {}
count_CB = {}
truck_intersection_frames = {}
truck_tracking_history = {}
vehicle_line_history = {}

def video_overlay(im0, w, h):
    # Line A
    line_a_start = (451, 736)
    line_a_end = (480, 381)
    cv2.line(im0, line_a_start, line_a_end, (0, 255, 0), thickness=2)
    
    # Line B
    line_b_start = (1374, 414)
    line_b_end = (1364, 760)
    cv2.line(im0, line_b_start, line_b_end, (255, 0, 0), thickness=2)
    
    # Line C
    line_c_start = (662, 219)
    line_c_end = (1238, 219)
    cv2.line(im0, line_c_start, line_c_end, (0, 0, 255), thickness=2)

    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    y_step = 20
    start_y = 20

    # Draw a smaller background rectangle just for the text (limit width to where text ends)
    cv2.rectangle(im0, (0, 0), (660, start_y + y_step * 6 + 10), (0, 0, 0), -1)

    columns = [
        {"title": "A to B:", "x": 10, "color": (0, 255, 0), "data": count_AB},
        {"title": "B to A:", "x": 170, "color": (255, 0, 0), "data": count_BA},
        {"title": "A to C:", "x": 330, "color": (0, 0, 255), "data": count_AC},
        {"title": "C to B:", "x": 490, "color": (255, 255, 0), "data": count_CB}
    ]

    for col in columns:
        x = col["x"]
        c_color = col["color"]
        data = col["data"]
        
        cv2.putText(im0, col["title"], (x, start_y), font, fontScale, c_color, thickness, cv2.LINE_AA)
        
        cars = data.get("Car", 0) + data.get("SUV", 0)
        van = data.get("Van", 0)
        light_truck = data.get("Truck", 0) + data.get("Pickup Truck", 0)
        heavy_truck = data.get("Heavy Truck", 0) + data.get("Machinery Vehicle", 0)
        bus = data.get("Bus", 0)
        motorcycle = data.get("Motorcycle", 0)

        cv2.putText(im0, f"Cars: {cars}", (x, start_y + y_step * 1), font, fontScale, c_color, thickness, cv2.LINE_AA)
        cv2.putText(im0, f"Van/MPV: {van}", (x, start_y + y_step * 2), font, fontScale, c_color, thickness, cv2.LINE_AA)
        cv2.putText(im0, f"Light truck: {light_truck}", (x, start_y + y_step * 3), font, fontScale, c_color, thickness, cv2.LINE_AA)
        cv2.putText(im0, f"Heavy truck: {heavy_truck}", (x, start_y + y_step * 4), font, fontScale, c_color, thickness, cv2.LINE_AA)
        cv2.putText(im0, f"Bus: {bus}", (x, start_y + y_step * 5), font, fontScale, c_color, thickness, cv2.LINE_AA)
        cv2.putText(im0, f"Motorcycle: {motorcycle}", (x, start_y + y_step * 6), font, fontScale, c_color, thickness, cv2.LINE_AA)


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')

    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    try:
        device = select_device(opt.device)
        use_cuda = device.type != 'cpu'
    except:
        print("GPU initialization failed, falling back to CPU")
        device = select_device('cpu')
        use_cuda = False
    
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=use_cuda)

    half &= device.type != 'cpu'

    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)
        os.makedirs(out)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)

    if opt.classes is None:
        opt.classes = [5, 34, 49, 65, 87, 199, 109, 55, 58]
    
    for c in opt.classes:
        count_AB[names[c]] = 0
        count_BA[names[c]] = 0
        count_AC[names[c]] = 0
        count_CB[names[c]] = 0
        sort_data[names[c]] = []

    half &= pt and device.type != 'cpu'
    if pt:
        model.model.half() if half else model.model.float()

    vid_path, vid_writer = None, None

    if show_vid:
        show_vid = check_imshow()

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = 1
    vid_path, vid_writer = [None] * bs, [None] * bs

    names = model.module.names if hasattr(model, 'module') else model.names

    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'gpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))

    break_loop = False

    global carAB, suvAB, motorcycleAB, truckAB, heavytruckAB, busAB, vanAB, pickupAB, machineryAB
    carAB, suvAB, motorcycleAB, truckAB, heavytruckAB, busAB, vanAB, pickupAB, machineryAB = 0,0,0,0,0,0,0,0,0
    global carBA, suvBA, motorcycleBA, truckBA, heavytruckBA, busBA, vanBA, pickupBA, machineryBA
    carBA, suvBA, motorcycleBA, truckBA, heavytruckBA, busBA, vanBA, pickupBA, machineryBA = 0,0,0,0,0,0,0,0,0
    global carAC, suvAC, motorcycleAC, truckAC, heavytruckAC, busAC, vanAC, pickupAC, machineryAC
    carAC, suvAC, motorcycleAC, truckAC, heavytruckAC, busAC, vanAC, pickupAC, machineryAC = 0,0,0,0,0,0,0,0,0
    global carCB, suvCB, motorcycleCB, truckCB, heavytruckCB, busCB, vanCB, pickupCB, machineryCB
    carCB, suvCB, motorcycleCB, truckCB, heavytruckCB, busCB, vanCB, pickupCB, machineryCB = 0,0,0,0,0,0,0,0,0

    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        if (frame_idx %1) == 0:    
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
            pred = model(img, augment=opt.augment, visualize=visualize)

            pred = non_max_suppression(pred, 0.01, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                    max_det=opt.max_det)

            for i, det in enumerate(pred):
                if webcam:
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)
                save_path = str(save_dir / p.name)
                s += '%gx%g ' % img.shape[2:]

                annotator = Annotator(im0, line_width=2, pil=False)

                w, h = im0.shape[1], im0.shape[0]

                current_detections = {}
                for c in opt.classes:
                    current_detections[names[c]] = 0

                if det is not None and len(det):
                    
                    mask = torch.ones(det.shape[0], dtype=torch.bool, device=det.device)
                    motorcycle_conf_thresh = 0.02
                    truck_conf_thresh = 0.05
                    bus_conf_thresh = 0.05
                    car_conf_thresh = 0.05
                    
                    for idx, (conf, cls) in enumerate(zip(det[:, 4], det[:, 5])):
                        if int(cls) == 58:  # Motorcycle
                            if conf < motorcycle_conf_thresh:
                                mask[idx] = False
                        elif int(cls) in [65, 87]:  # Truck and Pickup Truck
                            if conf < truck_conf_thresh:
                                mask[idx] = False
                        elif int(cls) in [199, 109]:  # Heavy Truck and Machinery Vehicle
                            if conf < truck_conf_thresh:
                                mask[idx] = False
                        elif int(cls) == 55:  # Bus
                            if conf < bus_conf_thresh:
                                mask[idx] = False
                        elif int(cls) in [5, 49, 34]:  # Car, Van, SUV (and others not matched above)
                            if conf < car_conf_thresh:
                                mask[idx] = False
                        else:
                            if conf < 0.05: # "rest 0.05"
                                mask[idx] = False
                    
                    det = det[mask]
                    
                    if len(det) == 0:
                        det = None

                if det is not None and len(det):

                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            current_detections[names[cls]] += 1
                            
                            truck_classes = ["Truck", "Pickup Truck", "Heavy Truck", "Machinery Vehicle"]
                            if names[cls] in truck_classes:
                                if id not in truck_tracking_history:
                                    truck_tracking_history[id] = []
                                truck_tracking_history[id].append(frame_idx)
                                if len(truck_tracking_history[id]) > 15:
                                    truck_tracking_history[id].pop(0)

                            count_obj(bboxes, w, id, names[cls], frame_idx)

                            c = int(cls)
                            label = f'{id} {names[c]}'
                            annotator.box_label(bboxes, label, color=colors(c, True))

                else:
                    deepsort.increment_ages()

                im0 = annotator.result()
                if show_vid:
                    video_overlay(im0, w, h)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):
                        raise StopIteration

                if break_loop:
                    break

                if save_vid:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = 25
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                    video_overlay(im0, w, h)
                    vid_writer.write(im0)

            frame_divider = 22500
            initial_row = 1
            excel = 'traffic_summary.xlsx'

            if (frame_idx%frame_divider) == 0 and frame_idx!=0:
                workbook = openpyxl.load_workbook(excel)
                sheet_name = 'lane_flow_counts'
                if sheet_name not in workbook.sheetnames:
                    workbook.create_sheet(sheet_name)
                sheet = workbook[sheet_name]
                row_counter = int(frame_idx/frame_divider)

                sheet.cell(row=initial_row, column=1).value = "AB"
                sheet.cell(row=initial_row+1, column=1).value = "1. Cars"
                sheet.cell(row=initial_row+row_counter+1, column=1).value = count_AB["Car"] - carAB + count_AB["SUV"] - suvAB
                sheet.cell(row=initial_row+1, column=2).value = "2. Van/MPV"
                sheet.cell(row=initial_row+row_counter+1, column=2).value = count_AB["Van"] - vanAB
                sheet.cell(row=initial_row+1, column=3).value = "3. Light truck"
                sheet.cell(row=initial_row+row_counter+1, column=3).value = count_AB["Truck"] - truckAB + count_AB["Pickup Truck"] - pickupAB
                sheet.cell(row=initial_row+1, column=4).value = "4. Heavy truck"
                sheet.cell(row=initial_row+row_counter+1, column=4).value = count_AB["Heavy Truck"] - heavytruckAB + count_AB["Machinery Vehicle"] - machineryAB
                sheet.cell(row=initial_row+1, column=5).value = "5. Bus"
                sheet.cell(row=initial_row+row_counter+1, column=5).value = count_AB["Bus"] - busAB
                sheet.cell(row=initial_row+1, column=6).value = "6. Motorcycle"
                sheet.cell(row=initial_row+row_counter+1, column=6).value = count_AB["Motorcycle"] - motorcycleAB

                sheet.cell(row=initial_row, column=8).value = "BA"
                sheet.cell(row=initial_row+1, column=8).value = "1. Cars"
                sheet.cell(row=initial_row+row_counter+1, column=8).value = count_BA["Car"] - carBA + count_BA["SUV"] - suvBA
                sheet.cell(row=initial_row+1, column=9).value = "2. Van/MPV"
                sheet.cell(row=initial_row+row_counter+1, column=9).value = count_BA["Van"] - vanBA
                sheet.cell(row=initial_row+1, column=10).value = "3. Light truck"
                sheet.cell(row=initial_row+row_counter+1, column=10).value = count_BA["Truck"] - truckBA + count_BA["Pickup Truck"] - pickupBA
                sheet.cell(row=initial_row+1, column=11).value = "4. Heavy truck"
                sheet.cell(row=initial_row+row_counter+1, column=11).value = count_BA["Heavy Truck"] - heavytruckBA + count_BA["Machinery Vehicle"] - machineryBA
                sheet.cell(row=initial_row+1, column=12).value = "5. Bus"
                sheet.cell(row=initial_row+row_counter+1, column=12).value = count_BA["Bus"] - busBA
                sheet.cell(row=initial_row+1, column=13).value = "6. Motorcycle"
                sheet.cell(row=initial_row+row_counter+1, column=13).value = count_BA["Motorcycle"] - motorcycleBA

                sheet.cell(row=initial_row, column=15).value = "AC"
                sheet.cell(row=initial_row+1, column=15).value = "1. Cars"
                sheet.cell(row=initial_row+row_counter+1, column=15).value = count_AC["Car"] - carAC + count_AC["SUV"] - suvAC
                sheet.cell(row=initial_row+1, column=16).value = "2. Van/MPV"
                sheet.cell(row=initial_row+row_counter+1, column=16).value = count_AC["Van"] - vanAC
                sheet.cell(row=initial_row+1, column=17).value = "3. Light truck"
                sheet.cell(row=initial_row+row_counter+1, column=17).value = count_AC["Truck"] - truckAC + count_AC["Pickup Truck"] - pickupAC
                sheet.cell(row=initial_row+1, column=18).value = "4. Heavy truck"
                sheet.cell(row=initial_row+row_counter+1, column=18).value = count_AC["Heavy Truck"] - heavytruckAC + count_AC["Machinery Vehicle"] - machineryAC
                sheet.cell(row=initial_row+1, column=19).value = "5. Bus"
                sheet.cell(row=initial_row+row_counter+1, column=19).value = count_AC["Bus"] - busAC
                sheet.cell(row=initial_row+1, column=20).value = "6. Motorcycle"
                sheet.cell(row=initial_row+row_counter+1, column=20).value = count_AC["Motorcycle"] - motorcycleAC

                sheet.cell(row=initial_row, column=22).value = "CB"
                sheet.cell(row=initial_row+1, column=22).value = "1. Cars"
                sheet.cell(row=initial_row+row_counter+1, column=22).value = count_CB["Car"] - carCB + count_CB["SUV"] - suvCB
                sheet.cell(row=initial_row+1, column=23).value = "2. Van/MPV"
                sheet.cell(row=initial_row+row_counter+1, column=23).value = count_CB["Van"] - vanCB
                sheet.cell(row=initial_row+1, column=24).value = "3. Light truck"
                sheet.cell(row=initial_row+row_counter+1, column=24).value = count_CB["Truck"] - truckCB + count_CB["Pickup Truck"] - pickupCB
                sheet.cell(row=initial_row+1, column=25).value = "4. Heavy truck"
                sheet.cell(row=initial_row+row_counter+1, column=25).value = count_CB["Heavy Truck"] - heavytruckCB + count_CB["Machinery Vehicle"] - machineryCB
                sheet.cell(row=initial_row+1, column=26).value = "5. Bus"
                sheet.cell(row=initial_row+row_counter+1, column=26).value = count_CB["Bus"] - busCB
                sheet.cell(row=initial_row+1, column=27).value = "6. Motorcycle"
                sheet.cell(row=initial_row+row_counter+1, column=27).value = count_CB["Motorcycle"] - motorcycleCB

                print(row_counter)
                
                carAB = count_AB["Car"]
                suvAB = count_AB["SUV"]
                motorcycleAB = count_AB["Motorcycle"]
                truckAB = count_AB["Truck"]
                heavytruckAB = count_AB["Heavy Truck"]
                busAB = count_AB["Bus"]
                vanAB = count_AB["Van"]
                pickupAB = count_AB["Pickup Truck"]
                machineryAB = count_AB["Machinery Vehicle"]
                
                carBA = count_BA["Car"]
                suvBA = count_BA["SUV"]
                motorcycleBA = count_BA["Motorcycle"]
                truckBA = count_BA["Truck"]
                heavytruckBA = count_BA["Heavy Truck"]
                busBA = count_BA["Bus"]
                vanBA = count_BA["Van"]
                pickupBA = count_BA["Pickup Truck"]
                machineryBA = count_BA["Machinery Vehicle"]

                carAC = count_AC["Car"]
                suvAC = count_AC["SUV"]
                motorcycleAC = count_AC["Motorcycle"]
                truckAC = count_AC["Truck"]
                heavytruckAC = count_AC["Heavy Truck"]
                busAC = count_AC["Bus"]
                vanAC = count_AC["Van"]
                pickupAC = count_AC["Pickup Truck"]
                machineryAC = count_AC["Machinery Vehicle"]

                carCB = count_CB["Car"]
                suvCB = count_CB["SUV"]
                motorcycleCB = count_CB["Motorcycle"]
                truckCB = count_CB["Truck"]
                heavytruckCB = count_CB["Heavy Truck"]
                busCB = count_CB["Bus"]
                vanCB = count_CB["Van"]
                pickupCB = count_CB["Pickup Truck"]
                machineryCB = count_CB["Machinery Vehicle"]

                workbook.save(excel)

    end_time = datetime.now()
    print('Elapsed time: {}'.format(end_time - start_time))


def count_obj(box, w, id, name, frame_idx):
    if id in ids:
        return
    
    left = int(box[0])
    top = int(box[1])
    right = int(box[2])
    bottom = int(box[3])
    
    # Line coordinates
    line_a_x1, line_a_y1 = 451, 736
    line_a_x2, line_a_y2 = 480, 381
    
    line_b_x1, line_b_y1 = 1374, 414
    line_b_x2, line_b_y2 = 1364, 760
    
    line_c_x1, line_c_y1 = 662, 219
    line_c_x2, line_c_y2 = 1238, 219
    
    truck_classes = ["Truck", "Pickup Truck", "Heavy Truck", "Machinery Vehicle"]
    bus_classes = ["Bus"]
    
    def line_intersects_box(x1, y1, x2, y2, box_left, box_top, box_right, box_bottom):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
        
        edges = [
            ((box_left, box_top), (box_right, box_top)),
            ((box_right, box_top), (box_right, box_bottom)),
            ((box_right, box_bottom), (box_left, box_bottom)),
            ((box_left, box_bottom), (box_left, box_top))
        ]
        
        for edge_start, edge_end in edges:
            if intersect((x1, y1), (x2, y2), edge_start, edge_end):
                return True
        return False
    
    # Check intersections with each line
    line_a_intersects = line_intersects_box(line_a_x1, line_a_y1, line_a_x2, line_a_y2, left, top, right, bottom)
    line_b_intersects = line_intersects_box(line_b_x1, line_b_y1, line_b_x2, line_b_y2, left, top, right, bottom)
    line_c_intersects = line_intersects_box(line_c_x1, line_c_y1, line_c_x2, line_c_y2, left, top, right, bottom)
    
    # Track which lines this vehicle has intersected
    if id not in vehicle_line_history:
        vehicle_line_history[id] = []
    
    # Record line intersections
    if line_a_intersects and 'A' not in vehicle_line_history[id]:
        vehicle_line_history[id].append('A')
    if line_b_intersects and 'B' not in vehicle_line_history[id]:
        vehicle_line_history[id].append('B')
    if line_c_intersects and 'C' not in vehicle_line_history[id]:
        vehicle_line_history[id].append('C')
    
    # Check for valid movement patterns and count
    if len(vehicle_line_history[id]) >= 2:
        first_line = vehicle_line_history[id][0]
        second_line = vehicle_line_history[id][1]
        
        # A to B movement
        if first_line == 'A' and second_line == 'B':
            if name in truck_classes:
                if id in truck_tracking_history and len(truck_tracking_history[id]) >= 5:
                    if id not in truck_intersection_frames:
                        truck_intersection_frames[id] = 0
                    truck_intersection_frames[id] += 1
                    if truck_intersection_frames[id] >= 5:
                        count_AB[name] += 1
                        ids.append(id)
                        print(id, " AB ", name)
                        del truck_intersection_frames[id]
                        if id in truck_tracking_history:
                            del truck_tracking_history[id]
            elif name in bus_classes:
                if id not in truck_intersection_frames:
                    truck_intersection_frames[id] = 0
                truck_intersection_frames[id] += 1
                if truck_intersection_frames[id] >= 5:
                    count_AB[name] += 1
                    ids.append(id)
                    print(id, " AB ", name)
                    del truck_intersection_frames[id]
            else:
                count_AB[name] += 1
                ids.append(id)
                print(id, " AB ", name)
        
        # B to A movement
        elif first_line == 'B' and second_line == 'A':
            if name in truck_classes:
                if id in truck_tracking_history and len(truck_tracking_history[id]) >= 5:
                    if id not in truck_intersection_frames:
                        truck_intersection_frames[id] = 0
                    truck_intersection_frames[id] += 1
                    if truck_intersection_frames[id] >= 5:
                        count_BA[name] += 1
                        ids.append(id)
                        print(id, " BA ", name)
                        del truck_intersection_frames[id]
                        if id in truck_tracking_history:
                            del truck_tracking_history[id]
            elif name in bus_classes:
                if id not in truck_intersection_frames:
                    truck_intersection_frames[id] = 0
                truck_intersection_frames[id] += 1
                if truck_intersection_frames[id] >= 5:
                    count_BA[name] += 1
                    ids.append(id)
                    print(id, " BA ", name)
                    del truck_intersection_frames[id]
            else:
                count_BA[name] += 1
                ids.append(id)
                print(id, " BA ", name)
        
        # A to C movement
        elif first_line == 'A' and second_line == 'C':
            if name in truck_classes:
                if id in truck_tracking_history and len(truck_tracking_history[id]) >= 5:
                    if id not in truck_intersection_frames:
                        truck_intersection_frames[id] = 0
                    truck_intersection_frames[id] += 1
                    if truck_intersection_frames[id] >= 5:
                        count_AC[name] += 1
                        ids.append(id)
                        print(id, " AC ", name)
                        del truck_intersection_frames[id]
                        if id in truck_tracking_history:
                            del truck_tracking_history[id]
            elif name in bus_classes:
                if id not in truck_intersection_frames:
                    truck_intersection_frames[id] = 0
                truck_intersection_frames[id] += 1
                if truck_intersection_frames[id] >= 5:
                    count_AC[name] += 1
                    ids.append(id)
                    print(id, " AC ", name)
                    del truck_intersection_frames[id]
            else:
                count_AC[name] += 1
                ids.append(id)
                print(id, " AC ", name)
        
        # C to B movement
        elif first_line == 'C' and second_line == 'B':
            if name in truck_classes:
                if id in truck_tracking_history and len(truck_tracking_history[id]) >= 5:
                    if id not in truck_intersection_frames:
                        truck_intersection_frames[id] = 0
                    truck_intersection_frames[id] += 1
                    if truck_intersection_frames[id] >= 5:
                        count_CB[name] += 1
                        ids.append(id)
                        print(id, " CB ", name)
                        del truck_intersection_frames[id]
                        if id in truck_tracking_history:
                            del truck_tracking_history[id]
            elif name in bus_classes:
                if id not in truck_intersection_frames:
                    truck_intersection_frames[id] = 0
                truck_intersection_frames[id] += 1
                if truck_intersection_frames[id] >= 5:
                    count_CB[name] += 1
                    ids.append(id)
                    print(id, " CB ", name)
                    del truck_intersection_frames[id]
            else:
                count_CB[name] += 1
                ids.append(id)
                print(id, " CB ", name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m_Objects365.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='videos/A0.mp4', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', default=True, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / '', help='save results to project/name')
    parser.add_argument('--name', default='traffic_export', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_false', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1

    with torch.no_grad():
        detect(opt)
