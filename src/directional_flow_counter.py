# limit the number of cpus used by high performance libraries
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

# from yolov5.models.experimental import attempt_load
# from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

green = int(550)
ids = []
blue = int(280)
sort_data = {}
sort_data_A = {}
# sort_data_B = {}
count_AB = {}
# count_BA = {}

def video_overlay(im0, w, h): # function to adjust video overlays like lines, counters etc
    start_point = (blue, 0)
    end_point = (blue, h)
    cv2.line(im0, start_point, end_point, (255, 0, 0), thickness=1) # blue

    start_point = (green, 0)
    end_point = (green, h)
    cv2.line(im0, start_point, end_point, (0, 255, 0), thickness=1) # green

    thickness = 1
    org = (10, 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.3

    cv2.rectangle(im0, (0, 0), (260, 105), (0, 0, 0), -1)    # black counter box

    cv2.putText(im0, "A to B:", org, font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

    for n, item in enumerate(count_AB.items()):
        cv2.putText(im0, str(item), (org[0], org[1] + 10 + n * 10), font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    # cv2.putText(im0, "B to A:", (org[0] + 120, org[1]), font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)

    # for n, item in enumerate(count_BA.items()):
    #     cv2.putText(im0, str(item), (org[0] + 120, org[1] + 10 + n * 10), font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    for c in opt.classes:
        count_AB[names[c]] = 0
        # count_BA[names[c]] = 0
        sort_data[names[c]] = []
        sort_data_A[names[c]] = []
        # sort_data_B[names[c]] = []

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Data loader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'gpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup

    break_loop = False

    global carAB, motorcycleAB, truckAB, heavytruckAB, busAB, suvAB, vanAB, pickupAB, machineryAB
    carAB, motorcycleAB, truckAB, heavytruckAB, busAB, suvAB, vanAB, pickupAB, machineryAB = 0,0,0,0,0,0,0,0,0
    # global carBA, motorcycleBA, truckBA, heavytruckBA, busBA, suvBA, vanBA, pickupBA, machineryBA
    # carBA, motorcycleBA, truckBA, heavytruckBA, busBA, suvBA, vanBA, pickupBA, machineryBA = 0,0,0,0,0,0,0,0,0

    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset): # start of frame-by-frame loop
        if (frame_idx %1) == 0:    
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
            pred = model(img, augment=opt.augment, visualize=visualize)

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                    max_det=opt.max_det)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                # seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                s += '%gx%g ' % img.shape[2:]  # print string

                annotator = Annotator(im0, line_width=2, pil=not ascii)

                w, h = im0.shape[1], im0.shape[0]

                current_detections = {}
                for c in opt.classes:
                    current_detections[names[c]] = 0

                if det is not None and len(det):

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deep sort
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            current_detections[names[cls]] += 1

                            count_obj(bboxes, w, id, names[cls])

                            c = int(cls)  # integer class
                            label = f'{id} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))

                else:
                    deepsort.increment_ages()

                # Stream results
                im0 = annotator.result()
                if show_vid:
                    video_overlay(im0, w, h)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                if break_loop:
                    break

                # Save results (image with detections)
                if save_vid:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            fps = 30
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                    video_overlay(im0, w, h)
                    vid_writer.write(im0)

            # write_to_excel(frame_idx)
            frame_divider = 27000 # how many frames per write to excel
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
                sheet.cell(row=initial_row+1, column=1).value = "Car"
                sheet.cell(row=initial_row+row_counter+1, column=1).value = count_AB["Car"] - carAB + count_AB["SUV"] - suvAB
                sheet.cell(row=initial_row+1, column=2).value = "Motorcycle"
                sheet.cell(row=initial_row+row_counter+1, column=2).value = count_AB["Motorcycle"] - motorcycleAB
                sheet.cell(row=initial_row+1, column=3).value = "Truck"
                sheet.cell(row=initial_row+row_counter+1, column=3).value = count_AB["Truck"] - truckAB + count_AB["Van"] - vanAB + count_AB["Pickup Truck"] - pickupAB
                sheet.cell(row=initial_row+1, column=4).value = "Heavy Truck"
                sheet.cell(row=initial_row+row_counter+1, column=4).value = count_AB["Heavy Truck"] - heavytruckAB + count_AB["Machinery Vehicle"] - machineryAB
                sheet.cell(row=initial_row+1, column=5).value = "Bus"
                sheet.cell(row=initial_row+row_counter+1, column=5).value = count_AB["Bus"] - busAB
                sheet.cell(row=initial_row+1, column=6).value = "SUV"
                sheet.cell(row=initial_row+row_counter+1, column=6).value = count_AB["SUV"] - suvAB
                sheet.cell(row=initial_row+1, column=7).value = "Van"
                sheet.cell(row=initial_row+row_counter+1, column=7).value = count_AB["Van"] - vanAB
                sheet.cell(row=initial_row+1, column=8).value = "Pickup Truck"
                sheet.cell(row=initial_row+row_counter+1, column=8).value = count_AB["Pickup Truck"] - pickupAB
                sheet.cell(row=initial_row+1, column=9).value = "Machinery Vehicle"
                sheet.cell(row=initial_row+row_counter+1, column=9).value = count_AB["Machinery Vehicle"] - machineryAB
                sheet.cell(row=initial_row+1, column=10).value = "Just Car"
                sheet.cell(row=initial_row+row_counter+1, column=10).value = count_AB["Car"] - carAB
                sheet.cell(row=initial_row+1, column=11).value = "Just Truck"
                sheet.cell(row=initial_row+row_counter+1, column=11).value = count_AB["Truck"] - truckAB
                sheet.cell(row=initial_row+1, column=12).value = "Just Heavy Truck"
                sheet.cell(row=initial_row+row_counter+1, column=12).value = count_AB["Heavy Truck"] - heavytruckAB

                
                # sheet.cell(row=initial_row, column=14).value = "BA"
                # sheet.cell(row=initial_row+1, column=14).value = "Car"
                # sheet.cell(row=initial_row+row_counter+1, column=14).value = count_BA["Car"] - carBA + count_BA["SUV"] - suvBA
                # sheet.cell(row=initial_row+1, column=15).value = "Motorcycle"
                # sheet.cell(row=initial_row+row_counter+1, column=15).value = count_BA["Motorcycle"] - motorcycleBA
                # sheet.cell(row=initial_row+1, column=16).value = "Truck"
                # sheet.cell(row=initial_row+row_counter+1, column=16).value = count_BA["Truck"] - truckBA + count_BA["Van"] - vanBA + count_BA["Pickup Truck"] - pickupBA
                # sheet.cell(row=initial_row+1, column=17).value = "Heavy Truck"
                # sheet.cell(row=initial_row+row_counter+1, column=17).value = count_BA["Heavy Truck"] - heavytruckBA + count_BA["Machinery Vehicle"] - machineryBA
                # sheet.cell(row=initial_row+1, column=18).value = "Bus"
                # sheet.cell(row=initial_row+row_counter+1, column=18).value = count_BA["Bus"] - busBA
                # sheet.cell(row=initial_row+1, column=19).value = "SUV"
                # sheet.cell(row=initial_row+row_counter+1, column=19).value = count_BA["SUV"] - suvBA
                # sheet.cell(row=initial_row+1, column=20).value = "Van"
                # sheet.cell(row=initial_row+row_counter+1, column=20).value = count_BA["Van"] - vanBA
                # sheet.cell(row=initial_row+1, column=21).value = "Pickup Truck"
                # sheet.cell(row=initial_row+row_counter+1, column=21).value = count_BA["Pickup Truck"] - pickupBA
                # sheet.cell(row=initial_row+1, column=22).value = "Machinery Vehicle"
                # sheet.cell(row=initial_row+row_counter+1, column=22).value = count_BA["Machinery Vehicle"] - machineryBA
                # sheet.cell(row=initial_row+1, column=23).value = "Just Car"
                # sheet.cell(row=initial_row+row_counter+1, column=23).value = count_BA["Car"] - carBA
                # sheet.cell(row=initial_row+1, column=24).value = "Just Truck"
                # sheet.cell(row=initial_row+row_counter+1, column=24).value = count_BA["Truck"] - truckBA
                # sheet.cell(row=initial_row+1, column=25).value = "Just Heavy Truck"
                # sheet.cell(row=initial_row+row_counter+1, column=25).value = count_BA["Heavy Truck"] - heavytruckBA
                print(row_counter)
                
                carAB = count_AB["Car"]
                motorcycleAB = count_AB["Motorcycle"]
                truckAB = count_AB["Truck"]
                heavytruckAB = count_AB["Heavy Truck"]
                busAB = count_AB["Bus"]
                suvAB = count_AB["SUV"]
                vanAB = count_AB["Van"]
                pickupAB = count_AB["Pickup Truck"]
                machineryAB = count_AB["Machinery Vehicle"]
                # carBA = count_BA["Car"]
                # motorcycleBA = count_BA["Motorcycle"]
                # truckBA = count_BA["Truck"]
                # heavytruckBA = count_BA["Heavy Truck"]
                # busBA = count_BA["Bus"]
                # suvBA = count_BA["SUV"]
                # vanBA = count_BA["Van"]
                # pickupBA = count_BA["Pickup Truck"]
                # machineryBA = count_BA["Machinery Vehicle"]

                workbook.save(excel)

    end_time = datetime.now()
    print('Elapsed time: {}'.format(end_time - start_time))


def count_obj(box, w, id, name):
    # left = int(box[0])
    # bottom = int(box[3])
    # right = int(box[2])
    # centroid_y = int(box[1] + (box[3] - box[1]) / 2)
    centroid_x = int(box[0] + (box[2] - box[0]) / 2)

    if id not in sort_data_A[name] and id not in sort_data[name]:
        # if name == "Motorcycle":
        #     count_AB[name] += 1
        #     sort_data_A[name].append(id)
        #     print(id," AB ",name)
        if green > centroid_x > blue:
            sort_data_A[name].append(id)
    
    # if id not in sort_data[name]:
    #     if centroid_x > blue:
    #         count_AB[name] += 1
    #         sort_data[name].append(id)
    #         print(id," AB ",name)
    #     elif centroid_x < blue:
    #         count_BA[name] += 1
    #         sort_data[name].append(id)
    #         print(id," BA ",name)
    
    if id in sort_data_A[name] and id not in sort_data[name] and id not in ids:
        # if green > centroid_x > blue:
        #     count_BA[name] += 1
        #     ids.append(id)
        #     print(id," BA ",name)
        if centroid_x < blue:
            count_AB[name] += 1
            sort_data[name].append(id)
            ids.append(id)
            print(id," AB ",name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m_Objects365.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='videos/A0.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # 5 car, 58 motorcycle, 65 truck, 199 heavy truck, 55 bus, 34 SUV, 49 van, 87 pickup truck, 109 machinery vehicle
    # (--classes 5 58 65 199 55 34 49 87 109)
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / '', help='save results to project/name')
    # parser.add_argument('--project', default='runs/traffic_flow', help='save results to project/name')
    parser.add_argument('--name', default='traffic_export', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_false', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
