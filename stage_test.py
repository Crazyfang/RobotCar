# coding: utf8
# Author: Qu shen

"""
Single grab single put in all area.
"""
from robot_car_handle_queue import RobotCarHandle
from shopping_robot.robot_control import RobotControl
import time

if __name__ == '__main__':
    # flodername = time.strftime()

    vision = RobotCarHandle()

    vision.open_camera()

    vision.second_process_pipe(0)

    # vision.second_process_queue(0)

    robot = RobotControl(shopping_list_path='./shopping_list.txt', use_box=True)

    for shelf in robot.shelf_list[0]:
        for i in range(1, 9):
            if i <= 6:
                robot.go_to(robot.camera_coordinate[shelf][2 * i], 'take picture')

            vision.camera_takephoto(i - 2)

    vision.image_handle_fixed_value()
    good_list = vision.return_first_result()

    for info in good_list:
        print(info)

        shelf_name, window, good_name = info

        robot.go_to(robot.window_coordinate[shelf_name][window], 'grab put')

        robot.grab_good(window, good_name)

    robot.go_to(robot.cart_position['yellow cube'], 'grab put')

    robot.put_into_cart('yellow cube')

    for j, shelf in enumerate(robot.shelf_list[:0:-1]):

        for i in range(1, 9):
            if i <= 6:
                robot.go_to(robot.camera_coordinate[shelf][14 - 2 * i], purpose='take picture')
            if i == 1 or i == 2:
                vision.camera_takephoto("temp")
            else:
                # D,C,B顺序版本
                vision.camera_takephoto(7 - (i - 2) + (2 - j + 1) * 6)
                vision.second_process_pipe(7 - (i - 2) + (2 - j + 1) * 6)

    vision.close_camera()

    vision.final_process_confirm_pipe()
    
    # vision.return_second_result()
    # vision.final_process_confirm()

    good_list = vision.return_second_result()

    if len(good_list) != 9:
        pass

    for info in good_list:
        shelf_name, window, good_name = info

        robot.go_to(robot.window_coordinate[shelf_name][window], 'grab put')

        robot.grab_good(window, good_name)

    robot.go_to(robot.cart_position['white pingpang ball'], 'grab put')

    robot.put_into_cart('white pingpang ball')

    robot.go_to((10,8), purpose='take picture')