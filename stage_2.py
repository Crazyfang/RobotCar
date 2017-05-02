from shopping_robot import ShoppingRobot
from Class_Feature_Recongnition import Main_Function
# TODO(qs): from  import

if __name__ == '__main__':

    # TODO(qs) vision =
    vision = Main_Function()
    vision.Open_Camera()
    robot = ShoppingRobot()

    for shelf in robot.shelf_list[0]:

        for i in range(1, 9):
            if i <= 6:
                robot.go_to(robot.window_coordinate[shelf][2 * i], purpose='take picture')

            vision.Camera_TakePhoto(i - 2)

        # vision.Main_Process()
        # good_list = vision.return_result()

        # for info in good_list:
        #     shelf_name, window, good_name = info
        #     robot.go_to(robot.window_coordinate[shelf_name][window], purpose='grab put')
        #
        #     robot.grab_good(window, good_name)
        #
        #     robot.go_to(robot.cart_position[good_name], purpose='grab put')
        #
        #     robot.put_in_cart(good_name)

    for j, shelf in enumerate(robot.shelf_list[-3:]):

        for i in range(1, 9):
            if i <= 6:
                robot.go_to(robot.window_coordinate[shelf][2 * i], purpose='take picture')
            if i == 1 or i == 2:
                vision.Camera_TakePhoto("temp")
            else:
                vision.Camera_TakePhoto(i + (j + 1) * 6 - 2)
                vision.Second_Process(i + (j + 1) * 6 - 2)

    vision.Close_Camera()
    vision.Final_Process()
    # vision.Dispose_Second()
    # good_list = vision.return_second_result()

    # for info in good_list:
    #     shelf_name, window, good_name = info
    #     robot.go_to(robot.window_coordinate[shelf_name][window], purpose='grab put')
    #
    #     robot.grab_good(window, good_name)
    #
    #     robot.go_to(robot.cart_position[good_name], purpose='grab put')
    #
    #     robot.put_in_cart(good_name)
