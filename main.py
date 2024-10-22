from kivy.clock import Clock
from kivy.lang import Builder
from kivy.config import Config
from kivy.logger import Logger
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager
from kivymd.uix.screen import MDScreen
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.menu import MDDropdownMenu
from kivymd.app import MDApp
from kivymd.toast import toast
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from pymodbus.client import ModbusTcpClient
from pymodbus.client import AsyncModbusTcpClient
from datetime import datetime
import matplotlib.pyplot as plt
import math as maths
import numpy as np
import time
import os
plt.style.use('bmh')

colors = {
    "Red": {
        "A200": "#EE2222",
        "A500": "#EE2222",
        "A700": "#EE2222",
    },

    "Gray": {
        "200": "#999999",
        "500": "#999999",
        "700": "#999999",
    },

    "Blue": {
        "200": "#196BA5",
        "500": "#196BA5",
        "700": "#196BA5",
    },

    "Light": {
        "StatusBar": "E0E0E0",
        "AppBar": "#202020",
        "Background": "#EEEEEE",
        "CardsDialogs": "#FFFFFF",
        "FlatButtonDown": "#CCCCCC",
    },

    "Dark": {
        "StatusBar": "101010",
        "AppBar": "#E0E0E0",
        "Background": "#111111",
        "CardsDialogs": "#000000",
        "FlatButtonDown": "#333333",
    },
}

modbus_client = ModbusTcpClient('192.168.1.111')

PULSE_PER_REV = 1600
GEAR_RATIO = 19

val_j1_pos_sv = 0.
val_j2_pos_sv = 0.
val_j3_pos_sv = 0.

val_j1_pos_pv = 0.
val_j2_pos_pv = 0.
val_j3_pos_pv = 0.

val_j1_vel = 30.
val_j2_vel = 30.
val_j3_vel = 30.

val_x_pos_sv = 0.
val_y_pos_sv = 0.
val_z_pos_sv = -700.

val_x_pos_pv = 0.
val_y_pos_pv = 0.
val_z_pos_pv = -700.

val_x_vel = 10.
val_y_vel = 10.
val_z_vel = 10.

val_x_step = np.zeros(10)
val_y_step = np.zeros(10)
val_z_step = np.zeros(10)
data_base_process = np.zeros([3, 10])

conf_x_speed_pv = 1
conf_y_speed_pv = 1
conf_z_speed_pv = 1
conf_bed_pos_pv = 0
conf_x_speed_sv = 1
conf_y_speed_sv = 1
conf_z_speed_sv = 1
conf_bed_pos_sv = 0
conf_x_speed_step = np.ones(10)
conf_y_speed_step = np.ones(10)
conf_z_speed_step = np.ones(10)
conf_bed_pos_step = np.zeros(10)
data_base_config = np.ones([4, 10])

flag_conn_stat = False
flag_mode = False
flag_run = False
flag_alarm = False
flag_reset = False

flag_jog_enable = False
flag_jog_req_x = False
flag_jog_req_y = False
flag_jog_req_z = False
flag_operate_req_x = False
flag_operate_req_y = False
flag_operate_req_z = False
flag_jog_req_j1 = False
flag_jog_req_j2 = False
flag_jog_req_j3 = False
flag_operate_req_j1 = False
flag_operate_req_j2 = False
flag_operate_req_j3 = False

flag_origin_req = False

flag_seqs_arr = np.zeros(11)
flag_steps_arr = np.zeros(11)

# ## versi TA
# LINK_JOINT_LENGTH = 295.0
# LINK_PARALLEL_LENGTH = 495.0
# JOINT_DISPLACEMENT = 230.0
# EFFECTOR_DISPLACEMENT = 80.0

tool_offset = 20.0

# versi Penelitian POLeLAND
LINK_JOINT_LENGTH = 640.0
LINK_PARALLEL_LENGTH = 840.0
JOINT_DISPLACEMENT = 225.7
EFFECTOR_DISPLACEMENT = 60.0

view_camera = np.array([45, 0, 0])

class DeltaPositionError(Exception):
    pass

class SimulatedDeltaBot(object):
    def __init__(self, LINK_JOINT_LENGTH, LINK_PARALLEL_LENGTH, JOINT_DISPLACEMENT, EFFECTOR_DISPLACEMENT):
        self.e = EFFECTOR_DISPLACEMENT
        self.f = JOINT_DISPLACEMENT
        self.re = LINK_PARALLEL_LENGTH
        self.rf = LINK_JOINT_LENGTH

    def forward(self, theta1, theta2, theta3):
        """ 
        Takes three servo angles in degrees.  Zero is horizontal.
        return (x,y,z) if point valid, None if not 
        """
        t = self.f-self.e

        theta1, theta2, theta3 = maths.radians(theta1), maths.radians(theta2), maths.radians(theta3)

        # Calculate position of leg1's joint.  x1 is implicitly zero - along the axis
        y1 = -(t + self.rf*maths.cos(theta1))
        z1 = -self.rf*maths.sin(theta1)

        # Calculate leg2's joint position
        y2 = (t + self.rf*maths.cos(theta2))*maths.sin(maths.pi/6)
        x2 = y2*maths.tan(maths.pi/3)
        z2 = -self.rf*maths.sin(theta2)

        # Calculate leg3's joint position
        y3 = (t + self.rf*maths.cos(theta3))*maths.sin(maths.pi/6)
        x3 = -y3*maths.tan(maths.pi/3)
        z3 = -self.rf*maths.sin(theta3)

        # From the three positions in space, determine if there is a valid
        # location for the effector
        dnm = (y2-y1)*x3-(y3-y1)*x2
    
        w1 = y1*y1 + z1*z1
        w2 = x2*x2 + y2*y2 + z2*z2
        w3 = x3*x3 + y3*y3 + z3*z3

        # x = (a1*z + b1)/dnm
        a1 = (z2-z1)*(y3-y1)-(z3-z1)*(y2-y1)
        b1 = -((w2-w1)*(y3-y1)-(w3-w1)*(y2-y1))/2.0

        # y = (a2*z + b2)/dnm;
        a2 = -(z2-z1)*x3+(z3-z1)*x2
        b2 = ((w2-w1)*x3 - (w3-w1)*x2)/2.0

        # a*z^2 + b*z + c = 0
        a = a1*a1 + a2*a2 + dnm*dnm
        b = 2*(a1*b1 + a2*(b2-y1*dnm) - z1*dnm*dnm)
        c = (b2-y1*dnm)*(b2-y1*dnm) + b1*b1 + dnm*dnm*(z1*z1 - self.re*self.re)
 
        # discriminant
        d = b*b - 4.0*a*c
        if d < 0:
            return None # non-existing point

        z0 = -0.5*(b+maths.sqrt(d))/a
        x0 = (a1*z0 + b1)/dnm
        y0 = (a2*z0 + b2)/dnm
        return (x0,y0,z0)

    def calculate_angle_yz(self, x0, y0, z0):
        y1 = -self.f
        y0 -= self.e
        a = (x0*x0 + y0*y0 + z0*z0 + self.rf*self.rf - self.re*self.re - y1*y1)/(2*z0)
        b = (y1-y0)/z0
        d = -(a + b*y1)*(a + b*y1) + self.rf*(b*b*self.rf + self.rf)
        if d < 0:
            raise DeltaPositionError()
        yj = (y1 - a*b - maths.sqrt(d))/(b*b + 1)
        zj = a + b*yj
        theta = 180.0*maths.atan(-zj/(y1-yj))/maths.pi
        if yj>y1:
            theta += 180.0
        return theta

    def inverse(self, x0, y0, z0):
        """
        Takes position and returns three servo angles, or 0,0,0 if not possible
        return (x,y,z) if point valid, None if not
        """
        cos120 = maths.cos(2.0*maths.pi/3.0)
        sin120 = maths.sin(2.0*maths.pi/3.0)

        try:
            theta1 = self.calculate_angle_yz(x0, y0, z0)
            theta2 = self.calculate_angle_yz(x0*cos120 + y0*sin120, y0*cos120 - x0*sin120, z0) # rotate +120 deg
            theta3 = self.calculate_angle_yz(x0*cos120 - y0*sin120, y0*cos120 + x0*sin120, z0) # rotate -120 deg

            return theta1, theta2, theta3
        except DeltaPositionError:
            print("error")
            return 0,0,0
        
class ScreenSplash(MDScreen):    
    def __init__(self, **kwargs):
        super(ScreenSplash, self).__init__(**kwargs)
        Clock.schedule_interval(self.update_progress_bar, .01)
        Clock.schedule_once(self.delayed_init, 5)
        
    def delayed_init(self, dt):
        Clock.schedule_interval(self.regular_update_connection, 5)
        Clock.schedule_interval(self.regular_display, 1)
        Clock.schedule_interval(self.regular_highspeed_display, 0.5)
        Clock.schedule_interval(self.regular_get_data, 0.5)

    def regular_update_connection(self, dt):
        global flag_conn_stat
        try:
            modbus_client.connect()
            flag_conn_stat = modbus_client.connected
            modbus_client.close()
        except Exception as e:
            toast(e)         

    def regular_get_data(self, dt):
        global flag_conn_stat, flag_mode, flag_run, flag_alarm, flag_reset, flag_jog_enable
        global val_x_pos_pv, val_y_pos_pv, val_z_pos_pv
        global val_x_pos_sv, val_y_pos_sv, val_z_pos_sv
        global val_j1_pos_pv, val_j2_pos_pv, val_j3_pos_pv
        global val_j1_pos_sv, val_j2_pos_sv, val_j3_pos_sv
        global flag_seqs_arr, flag_steps_arr

        try:
            if flag_conn_stat:
                modbus_client.connect()
                operate_flags = modbus_client.read_coils(3072, 4, slave=1) #M0 - M3
                pulse_registers = modbus_client.read_holding_registers(17500, 18, slave=1) #SV92 - SV109
                modbus_client.close()

                val_j1_pos_pv = pulse_registers.registers[3] * 360 / PULSE_PER_REV / GEAR_RATIO
                val_j2_pos_pv = pulse_registers.registers[9] * 360 / PULSE_PER_REV / GEAR_RATIO
                val_j3_pos_pv = pulse_registers.registers[15] * 360 / PULSE_PER_REV / GEAR_RATIO

                bot = SimulatedDeltaBot(LINK_JOINT_LENGTH, LINK_PARALLEL_LENGTH,
                                        JOINT_DISPLACEMENT, EFFECTOR_DISPLACEMENT)
                servo_angle = np.array([val_j1_pos_pv, val_j2_pos_pv, val_j3_pos_pv], dtype=int)

                fk_result = bot.forward(*servo_angle)
                
                val_x_pos_pv = fk_result[0]
                val_y_pos_pv = fk_result[1]
                val_z_pos_pv = fk_result[2]

                flag_mode = operate_flags.bits[0]
                flag_jog_enable = operate_flags.bits[1]

        except Exception as e:
            msg = f'{e}'
            toast(msg)  

    def regular_display(self, dt):
        global flag_conn_stat        
        global conf_bed_pos_step
        global val_x_pos_pv, val_y_pos_pv, val_z_pos_pv
        global val_j1_pos_pv, val_j2_pos_pv, val_j3_pos_pv

        try:
            screenMainMenu = self.screen_manager.get_screen('screen_main_menu')
            screenPipeSetting = self.screen_manager.get_screen('screen_pipe_setting')
            screenMachineSetting = self.screen_manager.get_screen('screen_machine_setting')
            screenAdvancedSetting = self.screen_manager.get_screen('screen_advanced_setting')
            screenOperateManual = self.screen_manager.get_screen('screen_operate_manual')
            screenOperateAuto = self.screen_manager.get_screen('screen_operate_auto')
            screenCompile = self.screen_manager.get_screen('screen_compile')

            if flag_conn_stat:
                screenMainMenu.ids.comm_status.text = "Status: Connected"
                screenMainMenu.ids.comm_status.color = "#196BA5"
                screenPipeSetting.ids.comm_status.text = "Status: Connected"
                screenPipeSetting.ids.comm_status.color = "#196BA5"
                screenMachineSetting.ids.comm_status.text = "Status: Connected"
                screenMachineSetting.ids.comm_status.color = "#196BA5"                        
                screenAdvancedSetting.ids.comm_status.text = "Status: Connected"
                screenAdvancedSetting.ids.comm_status.color = "#196BA5"  
                screenOperateManual.ids.comm_status.text = "Status: Connected"
                screenOperateManual.ids.comm_status.color = "#196BA5"  
                screenOperateAuto.ids.comm_status.text = "Status: Connected"
                screenOperateAuto.ids.comm_status.color = "#196BA5"  
                screenCompile.ids.comm_status.text = "Status: Connected"
                screenCompile.ids.comm_status.color = "#196BA5"  

                if conf_bed_pos_step[0] != 1:
                    screenCompile.ids.bt_bed_pos0.text = "DN"
                    screenCompile.ids.bt_bed_pos0.md_bg_color = "#196BA5"
                else:
                    screenCompile.ids.bt_bed_pos0.text = "UP"
                    screenCompile.ids.bt_bed_pos0.md_bg_color = "#ee2222"

                if conf_bed_pos_step[1] != 1:
                    screenCompile.ids.bt_bed_pos1.text = "DN"
                    screenCompile.ids.bt_bed_pos1.md_bg_color = "#196BA5"
                else:
                    screenCompile.ids.bt_bed_pos1.text = "UP"
                    screenCompile.ids.bt_bed_pos1.md_bg_color = "#ee2222"

                if conf_bed_pos_step[2] != 1:
                    screenCompile.ids.bt_bed_pos2.text = "DN"
                    screenCompile.ids.bt_bed_pos2.md_bg_color = "#196BA5"
                else:
                    screenCompile.ids.bt_bed_pos2.text = "UP"
                    screenCompile.ids.bt_bed_pos2.md_bg_color = "#ee2222"

                if conf_bed_pos_step[3] != 1:
                    screenCompile.ids.bt_bed_pos3.text = "DN"
                    screenCompile.ids.bt_bed_pos3.md_bg_color = "#196BA5"
                else:
                    screenCompile.ids.bt_bed_pos3.text = "UP"
                    screenCompile.ids.bt_bed_pos3.md_bg_color = "#ee2222"

                if conf_bed_pos_step[4] != 1:
                    screenCompile.ids.bt_bed_pos4.text = "DN"
                    screenCompile.ids.bt_bed_pos4.md_bg_color = "#196BA5"
                else:
                    screenCompile.ids.bt_bed_pos4.text = "UP"
                    screenCompile.ids.bt_bed_pos4.md_bg_color = "#ee2222"

                if conf_bed_pos_step[5] != 1:
                    screenCompile.ids.bt_bed_pos5.text = "DN"
                    screenCompile.ids.bt_bed_pos5.md_bg_color = "#196BA5"
                else:
                    screenCompile.ids.bt_bed_pos5.text = "UP"
                    screenCompile.ids.bt_bed_pos5.md_bg_color = "#ee2222"

                if conf_bed_pos_step[6] != 1:
                    screenCompile.ids.bt_bed_pos6.text = "DN"
                    screenCompile.ids.bt_bed_pos6.md_bg_color = "#196BA5"
                else:
                    screenCompile.ids.bt_bed_pos6.text = "UP"
                    screenCompile.ids.bt_bed_pos6.md_bg_color = "#ee2222"

                if conf_bed_pos_step[7] != 1:
                    screenCompile.ids.bt_bed_pos7.text = "DN"
                    screenCompile.ids.bt_bed_pos7.md_bg_color = "#196BA5"
                else:
                    screenCompile.ids.bt_bed_pos7.text = "UP"
                    screenCompile.ids.bt_bed_pos7.md_bg_color = "#ee2222"

                if conf_bed_pos_step[8] != 1:
                    screenCompile.ids.bt_bed_pos8.text = "DN"
                    screenCompile.ids.bt_bed_pos8.md_bg_color = "#196BA5"
                else:
                    screenCompile.ids.bt_bed_pos8.text = "UP"
                    screenCompile.ids.bt_bed_pos8.md_bg_color = "#ee2222"

                if conf_bed_pos_step[9] != 1:
                    screenCompile.ids.bt_bed_pos9.text = "DN"
                    screenCompile.ids.bt_bed_pos9.md_bg_color = "#196BA5"
                else:
                    screenCompile.ids.bt_bed_pos9.text = "UP"
                    screenCompile.ids.bt_bed_pos9.md_bg_color = "#ee2222"
                    
            else:
                screenMainMenu.ids.comm_status.text = "Status: Disconnected"
                screenMainMenu.ids.comm_status.color = "#ee2222"
                screenPipeSetting.ids.comm_status.text = "Status: Disconnected"
                screenPipeSetting.ids.comm_status.color = "#ee2222"
                screenMachineSetting.ids.comm_status.text = "Status: Disconnected"
                screenMachineSetting.ids.comm_status.color = "#ee2222"
                screenAdvancedSetting.ids.comm_status.text = "Status: Disconnected"
                screenAdvancedSetting.ids.comm_status.color = "#ee2222"
                screenOperateManual.ids.comm_status.text = "Status: Disconnected"
                screenOperateManual.ids.comm_status.color = "#ee2222"
                screenOperateAuto.ids.comm_status.text = "Status: Disconnected"
                screenOperateAuto.ids.comm_status.color = "#ee2222"
                screenCompile.ids.comm_status.text = "Status: Disconnected"
                screenCompile.ids.comm_status.color = "#ee2222"

        except Exception as e:
            Logger.error(e)

    def regular_highspeed_display(self, dt):
        global flag_mode, flag_run, flag_alarm
        global val_j1_pos_pv, val_j2_pos_pv, val_j3_pos_pv
        global val_x_pos_sv, val_y_pos_sv, val_z_pos_sv
        global flag_seqs_arr, flag_steps_arr

        screenOperateManual = self.screen_manager.get_screen('screen_operate_manual')
        screenOperateAuto = self.screen_manager.get_screen('screen_operate_auto')

        try:
            # screenOperateAuto.ids.lb_set_x.text = str(val_x_pos_sv)
            # screenOperateAuto.ids.lb_set_y.text = str(val_y_pos_sv)
            # screenOperateAuto.ids.lb_set_z.text = str(val_z_pos_sv)

            # screenOperateAuto.ids.lb_real_x.text = str(val_x_pv)
            # screenOperateAuto.ids.lb_real_y.text = str(val_y_pv)
            # screenOperateAuto.ids.lb_real_z.text = str(val_z_pv)

            # screenOperateAuto.ids.lb_x_speed.text = str(conf_x_speed_pv)
            # screenOperateAuto.ids.lb_y_speed.text = str(conf_y_speed_pv)
            # screenOperateAuto.ids.lb_z_speed.text = str(conf_z_speed_pv)
            # screenOperateAuto.ids.lb_bed_pos.text = "UP" if conf_bed_pos_pv == 1 else "DN"

            screenOperateManual.ids.lb_real_j1.text = f"{val_j1_pos_pv:.2f}"
            screenOperateManual.ids.lb_real_j2.text = f"{val_j2_pos_pv:.2f}"
            screenOperateManual.ids.lb_real_j3.text = f"{val_j3_pos_pv:.2f}"

            screenOperateManual.ids.lb_real_x.text = f"{val_x_pos_pv:.2f}"
            screenOperateManual.ids.lb_real_y.text = f"{val_y_pos_pv:.2f}"
            screenOperateManual.ids.lb_real_z.text = f"{val_z_pos_pv:.2f}"

            if not flag_mode:
                screenOperateManual.ids.bt_mode.md_bg_color = "#196BA5"
                screenOperateManual.ids.bt_mode.text = "MANUAL MODE"
                screenOperateAuto.ids.bt_mode.md_bg_color = "#196BA5"
                screenOperateAuto.ids.bt_mode.text = "MANUAL MODE"
            else:
                screenOperateManual.ids.bt_mode.md_bg_color = "#ee2222"
                screenOperateManual.ids.bt_mode.text = "AUTO MODE"
                screenOperateAuto.ids.bt_mode.md_bg_color = "#ee2222"
                screenOperateAuto.ids.bt_mode.text = "AUTO MODE"

            # if flag_run:
            #     screenOperateAuto.ids.lp_run.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_run.md_bg_color = "#223322"

            # if flag_alarm:
            #     screenOperateAuto.ids.lp_alarm.md_bg_color = "#ee2222"
            # else:
            #     screenOperateAuto.ids.lp_alarm.md_bg_color = "#332222"

            # if flag_seqs_arr[0]:
            #     screenOperateAuto.ids.lp_seq_init1.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_seq_init1.md_bg_color = "#223322"

            # if flag_seqs_arr[1]:
            #     screenOperateAuto.ids.lp_seq_init2.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_seq_init2.md_bg_color = "#223322"

            # if flag_seqs_arr[2]:
            #     screenOperateAuto.ids.lp_seq1.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_seq1.md_bg_color = "#223322"

            # if flag_seqs_arr[3]:
            #     screenOperateAuto.ids.lp_seq2.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_seq2.md_bg_color = "#223322"

            # if flag_seqs_arr[4]:
            #     screenOperateAuto.ids.lp_seq3.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_seq3.md_bg_color = "#223322"

            # if flag_seqs_arr[5]:
            #     screenOperateAuto.ids.lp_seq4.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_seq4.md_bg_color = "#223322"

            # if flag_seqs_arr[6]:
            #     screenOperateAuto.ids.lp_seq5.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_seq5.md_bg_color = "#223322"

            # if flag_seqs_arr[7]:
            #     screenOperateAuto.ids.lp_seq6.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_seq6.md_bg_color = "#223322"

            # if flag_seqs_arr[8]:
            #     screenOperateAuto.ids.lp_seq7.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_seq7.md_bg_color = "#223322"

            # if flag_seqs_arr[9]:
            #     screenOperateAuto.ids.lp_seq8.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_seq8.md_bg_color = "#223322"

            # if flag_seqs_arr[10]:
            #     screenOperateAuto.ids.lp_seq9.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_seq9.md_bg_color = "#223322"

            # if flag_steps_arr[0]:
            #     screenOperateAuto.ids.lp_step0.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_step0.md_bg_color = "#223322"

            # if flag_steps_arr[1]:
            #     screenOperateAuto.ids.lp_step1.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_step1.md_bg_color = "#223322"

            # if flag_steps_arr[2]:
            #     screenOperateAuto.ids.lp_step2.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_step2.md_bg_color = "#223322"

            # if flag_steps_arr[3]:
            #     screenOperateAuto.ids.lp_step3.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_step3.md_bg_color = "#223322"

            # if flag_steps_arr[4]:
            #     screenOperateAuto.ids.lp_step4.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_step4.md_bg_color = "#223322"

            # if flag_steps_arr[5]:
            #     screenOperateAuto.ids.lp_step5.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_step5.md_bg_color = "#223322"

            # if flag_steps_arr[6]:
            #     screenOperateAuto.ids.lp_step6.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_step6.md_bg_color = "#223322"

            # if flag_steps_arr[7]:
            #     screenOperateAuto.ids.lp_step7.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_step7.md_bg_color = "#223322"

            # if flag_steps_arr[8]:
            #     screenOperateAuto.ids.lp_step8.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_step8.md_bg_color = "#223322"

            # if flag_steps_arr[9]:
            #     screenOperateAuto.ids.lp_step9.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_step9.md_bg_color = "#223322"

            # if flag_steps_arr[10]:
            #     screenOperateAuto.ids.lp_step10.md_bg_color = "#22ee22"
            # else:
            #     screenOperateAuto.ids.lp_step10.md_bg_color = "#223322"

        except Exception as e:
            Logger.error(e)

    def update_progress_bar(self, *args):
        if (self.ids.progress_bar.value + 1) < 100:
            raw_value = self.ids.progress_bar_label.text.split('[')[-1]
            value = raw_value[:-2]
            value = eval(value.strip())
            new_value = value + 1
            self.ids.progress_bar.value = new_value
            self.ids.progress_bar_label.text = 'Loading.. [{:} %]'.format(new_value)
        else:
            self.ids.progress_bar.value = 100
            self.ids.progress_bar_label.text = 'Loading.. [{:} %]'.format(100)
            time.sleep(0.5)
            Clock.unschedule(self.update_progress_bar)
            self.screen_manager.current = 'screen_main_menu'
            return False
        
class ScreenMainMenu(MDScreen):    
    def __init__(self, **kwargs):
        super(ScreenMainMenu, self).__init__(**kwargs)

    def screen_main_menu(self):
        self.screen_manager.current = 'screen_main_menu'

    def screen_pipe_setting(self):
        self.screen_manager.current = 'screen_pipe_setting'

    def screen_machine_setting(self):
        self.screen_manager.current = 'screen_machine_setting'

    def screen_advanced_setting(self):
        self.screen_manager.current = 'screen_advanced_setting'

    def screen_operate_auto(self):
        self.screen_manager.current = 'screen_operate_manual'

    def screen_compile(self):
        self.screen_manager.current = 'screen_compile'

    def exec_shutdown(self):
        os.system("shutdown /s /t 1") #for windows os
        toast("shutting down system")
        # os.system("shutdown -h now")

class ScreenPipeSetting(MDScreen):
    def __init__(self, **kwargs):
        super(ScreenPipeSetting, self).__init__(**kwargs)
        Clock.schedule_once(self.delayed_init)

    def delayed_init(self, dt):
        self.load()

        self.update_graph()

    def update(self):
        global val_pipe_length
        global val_pipe_diameter
        global val_pipe_thickness

        val_pipe_length = float(self.ids.input_pipe_length.text)
        val_pipe_diameter = float(self.ids.input_pipe_diameter.text)
        val_pipe_thickness = float(self.ids.input_pipe_thickness.text)

        self.update_graph()

    def update_view(self, direction):
        global view_camera

        elev, azim, roll = view_camera
        
        if(direction == 0):
            print(elev)
            elev += 20

        if(direction == 1):
            print(elev)
            elev -= 20
        
        if(direction == 2):
            azim += 20
        
        if(direction == 3):
            azim -= 20
        
        view_camera = np.array([elev, azim, roll])        
        self.update_graph(elev, azim, roll)

    def update_graph(self, elev=45, azim=60, roll=0):
        global val_pipe_length
        global val_pipe_diameter
        global val_pipe_thickness
        global view_camera

        view_camera = elev, azim, roll

        try:
            self.ids.pipe_illustration.clear_widgets()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.fig.set_facecolor("#eeeeee")

            val_pipe_length = float(self.ids.input_pipe_length.text)
            val_pipe_diameter = float(self.ids.input_pipe_diameter.text)
            val_pipe_thickness = float(self.ids.input_pipe_thickness.text)

            Xr, Yr, Zr = self.simulate(val_pipe_length, val_pipe_diameter, val_pipe_thickness)

            self.ax.plot_surface(Xr, Yr, Zr, color='gray')
            self.ax.set_box_aspect(aspect=(1, 1, 1))
            self.ax.view_init(elev=view_camera[0], azim=view_camera[1], roll=view_camera[2])

            self.ids.pipe_illustration.add_widget(FigureCanvasKivyAgg(self.fig))
        except:
            toast("error update pipe illustration")

    def simulate(self, val_pipe_length, val_pipe_diameter, val_pipe_thickness):
        Uc = np.linspace(0, 2 * np.pi, 50)
        Xc = np.linspace(0, val_pipe_length, 2)

        Uc_inner = np.linspace(0, 2 * np.pi, 50)
        Xc_inner = np.linspace(0, val_pipe_length, 2)

        Uc, Xc = np.meshgrid(Uc, Xc)
        Uc_inner, Xc_inner = np.meshgrid(Uc_inner, Xc_inner)
        
        pipe_radius = val_pipe_diameter / 2
        pipe_radius_inner = (val_pipe_diameter / 2) - val_pipe_thickness

        Yc = pipe_radius * np.cos(Uc)
        Zc = pipe_radius * np.sin(Uc)

        Yc_inner = pipe_radius_inner * np.cos(Uc_inner)
        Zc_inner = pipe_radius_inner * np.sin(Uc_inner)

        Xr = np.append(Xc, Xc_inner, axis=0)
        Yr = np.append(Yc, Yc_inner, axis=0)
        Zr = np.append(Zc, Zc_inner, axis=0)

        Xr = np.append(Xr, Xc, axis=0)
        Yr = np.append(Yr, Yc, axis=0)
        Zr = np.append(Zr, Zc, axis=0)

        return Xr, Yr, Zr

    def load(self):
        global data_base_pipe_setting
        global val_pipe_length, val_pipe_diameter, val_pipe_thickness

        try:
            data_settings = np.loadtxt("conf\\settings.cfg", encoding=None)
            data_base_load = data_settings.T
            data_base_pipe_setting = data_base_load[:3]

            val_pipe_length = data_base_pipe_setting[0]
            val_pipe_diameter = data_base_pipe_setting[1]
            val_pipe_thickness = data_base_pipe_setting[2]

            self.ids.input_pipe_length.text = str(val_pipe_length)
            self.ids.input_pipe_diameter.text = str(val_pipe_diameter)
            self.ids.input_pipe_thickness.text = str(val_pipe_thickness)
            toast("sucessfully load pipe setting")
        except:
            toast("error load pipe setting")

    def save(self):
        global data_base_pipe_setting, data_base_machine_setting, data_base_advanced_setting
        global val_pipe_length, val_pipe_diameter, val_pipe_thickness

        try:
            self.update()

            data_base_pipe_setting = np.array([val_pipe_length,
                                   val_pipe_diameter,
                                   val_pipe_thickness])

            data_base_save = np.hstack((data_base_pipe_setting, data_base_machine_setting, data_base_advanced_setting))
            with open("conf\\settings.cfg","wb") as f:
                np.savetxt(f, data_base_save.T, fmt="%.3f")
            toast("sucessfully save pipe setting")
        except:
            toast("error save pipe setting")

    def menu_callback(self, text_item):
        print(text_item)

    def screen_main_menu(self):
        self.screen_manager.current = 'screen_main_menu'

    def screen_pipe_setting(self):
        self.screen_manager.current = 'screen_pipe_setting'

    def screen_machine_setting(self):
        self.screen_manager.current = 'screen_machine_setting'

    def screen_advanced_setting(self):
        self.screen_manager.current = 'screen_advanced_setting'

    def screen_operate_auto(self):
        self.screen_manager.current = 'screen_operate_auto'

    def screen_compile(self):
        self.screen_manager.current = 'screen_compile'

    def exec_shutdown(self):
        os.system("shutdown /s /t 1") #for windows os
        toast("shutting down system")
        # os.system("shutdown -h 1")

class ScreenMachineSetting(MDScreen):
    def __init__(self, **kwargs):
        super(ScreenMachineSetting, self).__init__(**kwargs)
        Clock.schedule_once(self.delayed_init)

    def delayed_init(self, dt):
        self.load()

    def update(self):
        global flag_conn_stat

        global val_machine_eff_length
        global val_machine_supp_pos
        global val_machine_clamp_front_delay
        global val_machine_clamp_rear_delay
        global val_machine_press_front_delay
        global val_machine_press_rear_delay
        global val_machine_collet_clamp_delay
        global val_machine_collet_open_delay
        global val_machine_die_radius

        val_machine_eff_length = float(self.ids.input_machine_eff_length.text)
        val_machine_supp_pos = float(self.ids.input_machine_supp_pos.text)
        val_machine_clamp_front_delay = float(self.ids.input_machine_clamp_front_delay.text)
        val_machine_clamp_rear_delay = float(self.ids.input_machine_clamp_rear_delay.text)
        val_machine_press_front_delay = float(self.ids.input_machine_press_front_delay.text)
        val_machine_press_rear_delay = float(self.ids.input_machine_press_rear_delay.text)
        val_machine_collet_clamp_delay = float(self.ids.input_machine_collet_clamp_delay.text)
        val_machine_collet_open_delay = float(self.ids.input_machine_collet_open_delay.text)
        val_machine_die_radius = float(self.ids.input_machine_die_radius.text)

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_register(2512, int(val_machine_eff_length), slave=1) #V2000
                modbus_client.write_register(2513, int(val_machine_supp_pos), slave=1) #V2001
                modbus_client.write_register(2514, int(val_machine_clamp_front_delay), slave=1) #V2002
                modbus_client.write_register(2515, int(val_machine_clamp_rear_delay), slave=1) #V2003
                modbus_client.write_register(2516, int(val_machine_press_front_delay), slave=1) #V2004
                modbus_client.write_register(2517, int(val_machine_press_rear_delay), slave=1) #V2005
                modbus_client.write_register(2518, int(val_machine_collet_clamp_delay), slave=1) #V2006
                modbus_client.write_register(2519, int(val_machine_collet_open_delay), slave=1) #V2007
                modbus_client.write_register(2520, int(val_machine_die_radius), slave=1) #V2008
                modbus_client.close()
            else:
                toast("PLC Slave is not connected")  

        except:
            toast("error send machine_setting data to PLC Slave")    

    def update_image(self, image_num):
        if image_num == 0:
            self.ids.machine_image.source = 'asset/machine_setting_eff_length.png'
        elif image_num == 1:
            self.ids.machine_image.source = 'asset/machine_setting_supp_pos.png'
        elif image_num == 2:
            self.ids.machine_image.source = 'asset/machine_setting_clamp_front_delay.png'
        elif image_num == 3:
            self.ids.machine_image.source = 'asset/machine_setting_clamp_rear_delay.png'
        elif image_num == 4:
            self.ids.machine_image.source = 'asset/machine_setting_press_front_delay.png'
        elif image_num == 5:
            self.ids.machine_image.source = 'asset/machine_setting_press_rear_delay.png'
        elif image_num == 6:
            self.ids.machine_image.source = 'asset/machine_setting_collet_clamp_delay.png'
        elif image_num == 7:
            self.ids.machine_image.source = 'asset/machine_setting_collet_open_delay.png'

    def load(self):
        global data_base_machine_setting
        global val_machine_eff_length, val_machine_supp_pos, val_machine_clamp_front_delay, val_machine_clamp_rear_delay
        global val_machine_press_front_delay, val_machine_press_rear_delay, val_machine_collet_clamp_delay
        global val_machine_collet_open_delay, val_machine_die_radius

        try:
            data_settings = np.loadtxt("conf\\settings.cfg", encoding=None)
            data_base_load = data_settings.T
            data_base_machine_setting = data_base_load[3:12]

            val_machine_eff_length = data_base_machine_setting[0]
            val_machine_supp_pos = data_base_machine_setting[1]
            val_machine_clamp_front_delay = data_base_machine_setting[2]
            val_machine_clamp_rear_delay = data_base_machine_setting[3]
            val_machine_press_front_delay = data_base_machine_setting[4]
            val_machine_press_rear_delay = data_base_machine_setting[5]
            val_machine_collet_clamp_delay = data_base_machine_setting[6]
            val_machine_collet_open_delay = data_base_machine_setting[7]
            val_machine_die_radius = data_base_machine_setting[8]

            self.ids.input_machine_eff_length.text = str(val_machine_eff_length)
            self.ids.input_machine_supp_pos.text = str(val_machine_supp_pos)
            self.ids.input_machine_clamp_front_delay.text = str(val_machine_clamp_front_delay)
            self.ids.input_machine_clamp_rear_delay.text = str(val_machine_clamp_rear_delay)
            self.ids.input_machine_press_front_delay.text = str(val_machine_press_front_delay)
            self.ids.input_machine_press_rear_delay.text = str(val_machine_press_rear_delay)
            self.ids.input_machine_collet_clamp_delay.text = str(val_machine_collet_clamp_delay)
            self.ids.input_machine_collet_open_delay.text = str(val_machine_collet_open_delay)
            self.ids.input_machine_die_radius.text = str(val_machine_die_radius)
            toast("sucessfully load machine setting")
        except:
            toast("error load machine setting")

    def save(self):
        global data_base_pipe_setting, data_base_machine_setting, data_base_advanced_setting
        global val_machine_eff_length, val_machine_supp_pos, val_machine_clamp_front_delay, val_machine_clamp_rear_delay
        global val_machine_press_front_delay, val_machine_press_rear_delay, val_machine_collet_clamp_delay
        global val_machine_collet_open_delay, val_machine_die_radius

        try:
            self.update()
            
            data_base_machine_setting = np.array([val_machine_eff_length,
                                   val_machine_supp_pos,
                                   val_machine_clamp_front_delay,
                                   val_machine_clamp_rear_delay,
                                   val_machine_press_front_delay,
                                   val_machine_press_rear_delay,
                                   val_machine_collet_clamp_delay,
                                   val_machine_collet_open_delay,
                                   val_machine_die_radius,
                                   ])
            
            data_base_save = np.hstack((data_base_pipe_setting, data_base_machine_setting, data_base_advanced_setting))
            with open("conf\\settings.cfg","wb") as f:
                np.savetxt(f, data_base_save.T, fmt="%.3f")
            toast("sucessfully save machine setting")
        except:
            toast("error save machine setting")

    def screen_main_menu(self):
        self.screen_manager.current = 'screen_main_menu'

    def screen_pipe_setting(self):
        self.screen_manager.current = 'screen_pipe_setting'

    def screen_machine_setting(self):
        self.screen_manager.current = 'screen_machine_setting'

    def screen_advanced_setting(self):
        self.screen_manager.current = 'screen_advanced_setting'

    def screen_operate_auto(self):
        self.screen_manager.current = 'screen_operate_auto'

    def screen_compile(self):
        self.screen_manager.current = 'screen_compile'

    def exec_shutdown(self):
        os.system("shutdown /s /t 1") #for windows os
        toast("shutting down system")
        # os.system("shutdown -h 1")

class ScreenAdvancedSetting(MDScreen):
    def __init__(self, **kwargs):
        super(ScreenAdvancedSetting, self).__init__(**kwargs)
        Clock.schedule_once(self.delayed_init)

    def delayed_init(self, dt):
        self.load()

    def update(self):
        global modbus_client

        global val_advanced_pipe_head
        global val_advanced_start_mode
        global val_advanced_first_line
        global val_advanced_finish_job
        global val_advanced_receive_pos_x
        global val_advanced_receive_pos_b
        global val_advanced_prod_qty
        global val_advanced_press_semiclamp_time
        global val_advanced_press_semiopen_time
        global val_advanced_clamp_semiclamp_time
        global val_advanced_springback_20
        global val_advanced_springback_120
        global val_advanced_max_y
        global val_advanced_press_start_angle
        global val_advanced_press_stop_angle

        val_advanced_pipe_head = float(self.ids.input_advanced_pipe_head.text)
        val_advanced_start_mode = float(self.ids.input_advanced_start_mode.text)
        val_advanced_first_line = float(self.ids.input_advanced_first_line.text)
        val_advanced_finish_job = float(self.ids.input_advanced_finish_job.text)
        val_advanced_receive_pos_x = float(self.ids.input_advanced_receive_pos_x.text)
        val_advanced_receive_pos_b = float(self.ids.input_advanced_receive_pos_b.text)
        val_advanced_prod_qty = float(self.ids.input_advanced_prod_qty.text)
        val_advanced_press_semiclamp_time = float(self.ids.input_advanced_press_semiclamp_time.text)
        val_advanced_press_semiopen_time = float(self.ids.input_advanced_press_semiopen_time.text)
        val_advanced_clamp_semiclamp_time = float(self.ids.input_advanced_clamp_semiclamp_time.text)
        val_advanced_springback_20 = float(self.ids.input_advanced_springback_20.text)
        val_advanced_springback_120 = float(self.ids.input_advanced_springback_120.text)
        val_advanced_max_y = float(self.ids.input_advanced_max_y.text)
        val_advanced_press_start_angle = float(self.ids.input_advanced_press_start_angle.text)
        val_advanced_press_stop_angle = float(self.ids.input_advanced_press_stop_angle.text)

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_register(2522, int(val_advanced_pipe_head), slave=1) #V2010
                modbus_client.write_register(2523, int(val_advanced_start_mode), slave=1) #V2011
                modbus_client.write_register(2524, int(val_advanced_first_line), slave=1) #V2012
                modbus_client.write_register(2525, int(val_advanced_finish_job), slave=1) #V2013
                modbus_client.write_register(2526, int(val_advanced_receive_pos_x), slave=1) #V2014
                modbus_client.write_register(2527, int(val_advanced_receive_pos_b), slave=1) #V2015
                modbus_client.write_register(2528, int(val_advanced_prod_qty), slave=1) #V2016
                modbus_client.write_register(2529, int(val_advanced_press_semiclamp_time), slave=1) #V2017
                modbus_client.write_register(2530, int(val_advanced_press_semiopen_time), slave=1) #V2018
                modbus_client.write_register(2531, int(val_advanced_clamp_semiclamp_time), slave=1) #V2019
                modbus_client.write_register(2532, int(val_advanced_springback_20), slave=1) #V2020
                modbus_client.write_register(2533, int(val_advanced_springback_120), slave=1) #V2021
                modbus_client.write_register(2534, int(val_advanced_max_y), slave=1) #V2022
                modbus_client.write_register(2535, int(val_advanced_press_start_angle), slave=1) #V2023
                modbus_client.write_register(2536, int(val_advanced_press_stop_angle), slave=1) #V2024
                modbus_client.close()
            else:
                toast("PLC Slave is not connected")  
        except:
            toast("error send machine_setting data to PLC Slave") 

    def load(self):
        global data_base_advanced_setting
        global val_advanced_pipe_head, val_advanced_start_mode, val_advanced_first_line, val_advanced_finish_job
        global val_advanced_receive_pos_x, val_advanced_receive_pos_b, val_advanced_prod_qty, val_advanced_press_semiclamp_time
        global val_advanced_press_semiopen_time, val_advanced_clamp_semiclamp_time, val_advanced_springback_20, val_advanced_springback_120
        global val_advanced_max_y, val_advanced_press_start_angle, val_advanced_press_stop_angle

        try:
            data_settings = np.loadtxt("conf\\settings.cfg", encoding=None)
            data_base_load = data_settings.T
            data_base_advanced_setting = data_base_load[12:]

            val_advanced_pipe_head = data_base_advanced_setting[0]
            val_advanced_start_mode = data_base_advanced_setting[1]
            val_advanced_first_line = data_base_advanced_setting[2]
            val_advanced_finish_job = data_base_advanced_setting[3]
            val_advanced_receive_pos_x = data_base_advanced_setting[4]
            val_advanced_receive_pos_b = data_base_advanced_setting[5]
            val_advanced_prod_qty = data_base_advanced_setting[6]
            val_advanced_press_semiclamp_time = data_base_advanced_setting[7]
            val_advanced_press_semiopen_time = data_base_advanced_setting[8]
            val_advanced_clamp_semiclamp_time = data_base_advanced_setting[9]
            val_advanced_springback_20 = data_base_advanced_setting[10]
            val_advanced_springback_120 = data_base_advanced_setting[11]
            val_advanced_max_y = data_base_advanced_setting[12]
            val_advanced_press_start_angle = data_base_advanced_setting[13]
            val_advanced_press_stop_angle = data_base_advanced_setting[14]

            self.ids.input_advanced_pipe_head.text = str(val_advanced_pipe_head)
            self.ids.input_advanced_start_mode.text = str(val_advanced_start_mode)
            self.ids.input_advanced_first_line.text = str(val_advanced_first_line)
            self.ids.input_advanced_finish_job.text = str(val_advanced_finish_job)
            self.ids.input_advanced_receive_pos_x.text = str(val_advanced_receive_pos_x)
            self.ids.input_advanced_receive_pos_b.text = str(val_advanced_receive_pos_b)
            self.ids.input_advanced_prod_qty.text = str(val_advanced_prod_qty)
            self.ids.input_advanced_press_semiclamp_time.text = str(val_advanced_press_semiclamp_time)
            self.ids.input_advanced_press_semiopen_time.text = str(val_advanced_press_semiopen_time)
            self.ids.input_advanced_clamp_semiclamp_time.text = str(val_advanced_clamp_semiclamp_time)
            self.ids.input_advanced_springback_20.text = str(val_advanced_springback_20)
            self.ids.input_advanced_springback_120.text = str(val_advanced_springback_120)
            self.ids.input_advanced_max_y.text = str(val_advanced_max_y)
            self.ids.input_advanced_press_start_angle.text = str(val_advanced_press_start_angle)
            self.ids.input_advanced_press_stop_angle.text = str(val_advanced_press_stop_angle)
            toast("sucessfully load advanced setting")
        except:
            toast("error load advanced setting")

    def save(self):
        global data_base_pipe_setting, data_base_machine_setting, data_base_advanced_setting
        global val_machine_eff_length, val_machine_supp_pos, val_machine_clamp_front_delay, val_machine_clamp_rear_delay
        global val_machine_press_front_delay, val_machine_press_rear_delay, val_machine_collet_clamp_delay
        global val_machine_collet_open_delay, val_machine_die_radius

        try:
            self.update()
            
            data_base_advanced_setting = np.array([val_advanced_pipe_head,
                                   val_advanced_start_mode,
                                   val_advanced_first_line,
                                   val_advanced_finish_job,
                                   val_advanced_receive_pos_x,
                                   val_advanced_receive_pos_b,
                                   val_advanced_prod_qty,
                                   val_advanced_press_semiclamp_time,
                                   val_advanced_press_semiopen_time,
                                   val_advanced_clamp_semiclamp_time,
                                   val_advanced_springback_20,
                                   val_advanced_springback_120,
                                   val_advanced_max_y,
                                   val_advanced_press_start_angle,
                                   val_advanced_press_stop_angle,
                                   ])
            
            data_base_save = np.hstack((data_base_pipe_setting, data_base_machine_setting, data_base_advanced_setting))
            with open("conf\\settings.cfg","wb") as f:
                np.savetxt(f, data_base_save.T, fmt="%.3f")
            toast("sucessfully save advanced setting")
        except:
            toast("error save advanced setting")

    def screen_main_menu(self):
        self.screen_manager.current = 'screen_main_menu'

    def screen_pipe_setting(self):
        self.screen_manager.current = 'screen_pipe_setting'

    def screen_machine_setting(self):
        self.screen_manager.current = 'screen_machine_setting'

    def screen_advanced_setting(self):
        self.screen_manager.current = 'screen_advanced_setting'

    def screen_operate_auto(self):
        self.screen_manager.current = 'screen_operate_auto'

    def screen_compile(self):
        self.screen_manager.current = 'screen_compile'

    def exec_shutdown(self):
        os.system("shutdown /s /t 1") #for windows os
        toast("shutting down system")
        # os.system("shutdown -h 1")

class ScreenOperateManual(MDScreen):
    def __init__(self, **kwargs):      
        super(ScreenOperateManual, self).__init__(**kwargs)
        Clock.schedule_once(self.delayed_init, 5)

    def delayed_init(self, dt):
        self.ids.input_operate_x.text = str(val_x_pos_sv)
        self.ids.input_operate_y.text = str(val_y_pos_sv)
        self.ids.input_operate_z.text = str(val_z_pos_sv)
        self.ids.input_operate_j1.text = str(val_j1_pos_sv)
        self.ids.input_operate_j2.text = str(val_j2_pos_sv)
        self.ids.input_operate_j3.text = str(val_j3_pos_sv)
        self.ids.input_vel_j1.text = str(val_j1_vel)
        self.ids.input_vel_j2.text = str(val_j2_vel)
        self.ids.input_vel_j3.text = str(val_j3_vel)        
        self.reload()

    def update_view(self, direction):
        global view_camera
        elev, azim, roll = view_camera
        
        if(direction == 0):
            print(elev)
            elev += 20

        if(direction == 1):
            print(elev)
            elev -= 20
        
        if(direction == 2):
            azim += 20
        
        if(direction == 3):
            azim -= 20
        
        view_camera = np.array([elev, azim, roll])        
        self.update_graph(elev, azim, roll)

    def reload(self):
        global data_base_process
        
        self.update_graph()
            
    def update_graph(self, elev=45, azim=60, roll=0):
        global val_x_step
        global val_y_step
        global val_z_step
        global val_x_pos_sv, val_y_pos_sv, val_z_pos_sv

        global data_base_process
        view_camera = elev, azim, roll
        try:
            self.ids.delta_robot_illustration.clear_widgets()

            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.fig.set_facecolor("#eeeeee")

            bot = SimulatedDeltaBot(LINK_JOINT_LENGTH, LINK_PARALLEL_LENGTH,
                                    JOINT_DISPLACEMENT, EFFECTOR_DISPLACEMENT)
            # robot modelling
            # calculate IK for control system, input destined coordinate, output joint angle 
            ik_result = bot.inverse(val_x_pos_sv ,val_y_pos_sv, val_z_pos_sv)
            print(ik_result)

            servo_angle = np.array([ik_result[0],ik_result[1],ik_result[2]], dtype=int)

            cos120 = maths.cos(2.0*maths.pi/3.0)
            sin120 = maths.sin(2.0*maths.pi/3.0)

            fk_result = bot.forward(*servo_angle)
            print(fk_result)

            base = np.array([[0, -JOINT_DISPLACEMENT, 0],
                    [sin120*JOINT_DISPLACEMENT,-cos120*JOINT_DISPLACEMENT,0],
                    [-sin120*JOINT_DISPLACEMENT,-cos120*JOINT_DISPLACEMENT,0]])
            
            platform = np.array([[fk_result[0], fk_result[1]-EFFECTOR_DISPLACEMENT,fk_result[2]],
                    [fk_result[0]+sin120*EFFECTOR_DISPLACEMENT,fk_result[1]-cos120*EFFECTOR_DISPLACEMENT,fk_result[2]],
                    [fk_result[0]-sin120*EFFECTOR_DISPLACEMENT,fk_result[1]-cos120*EFFECTOR_DISPLACEMENT,fk_result[2]]])
            
            t = JOINT_DISPLACEMENT-EFFECTOR_DISPLACEMENT
            theta1, theta2, theta3 = maths.radians(servo_angle[0]), maths.radians(servo_angle[1]), maths.radians(servo_angle[2])
            # Calculate position of leg1's joint.  x1 is implicitly zero - along the axis
            y1 = -(t + LINK_JOINT_LENGTH*maths.cos(theta1))
            z1 = -LINK_JOINT_LENGTH*maths.sin(theta1)
            # Calculate leg2's joint position
            y2 = (t + LINK_JOINT_LENGTH*maths.cos(theta2))*maths.sin(maths.pi/6)
            x2 = y2*maths.tan(maths.pi/3)
            z2 = -LINK_JOINT_LENGTH*maths.sin(theta2)
            # Calculate leg3's joint position
            y3 = (t + LINK_JOINT_LENGTH*maths.cos(theta3))*maths.sin(maths.pi/6)
            x3 = -y3*maths.tan(maths.pi/3)
            z3 = -LINK_JOINT_LENGTH*maths.sin(theta3)

            joint = np.array([[0,y1,z1],
                    [x2,y2,z2],
                    [x3,y3,z3]])
            
            self.ax.scatter(xs=[x for x,y,z in base] ,ys=[y for x,y,z in base],zs=[z for x,y,z in base])
            self.ax.scatter(xs=[x for x,y,z in platform] ,ys=[y for x,y,z in platform],zs=[z for x,y,z in platform])
            
            for i in range(3):
                self.ax.plot([base.T[0,i] ,joint.T[0,i]],[base.T[1,i],joint.T[1,i]],[base.T[2,i],joint.T[2,i]])
            for i in range(3):
                self.ax.plot([joint.T[0,i] ,platform.T[0,i]],[joint.T[1,i],platform.T[1,i]],[joint.T[2,i],platform.T[2,i]])            
            
            self.ax.set_box_aspect(aspect=(1, 1, 1))
            # self.ax.set_aspect('equal')

            self.ax.set_xlim([-800, 800])
            self.ax.set_ylim([-800, 800])
            self.ax.set_zlim([-1400, 0])
            # self.ax.axis('off')
            self.ax.view_init(elev=view_camera[0], azim=view_camera[1], roll=view_camera[2])
            self.ids.delta_robot_illustration.add_widget(FigureCanvasKivyAgg(self.fig))   
        except:
            toast("error update pipe ying process illustration")
           
    def sign_int(self, value):
        value = int(value)
        if value < 0:
            return value + (2**16)
        else:
            return value
        
    def exec_mode(self):
        global flag_conn_stat, flag_mode

        if flag_mode:
            flag_mode = False
        else:
            flag_mode = True

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3072, flag_mode, slave=1) #M0
                modbus_client.close()
        except Exception as e:
            toast(e) 

    def exec_jog_enable(self):
        global flag_conn_stat, flag_jog_enable
        if flag_jog_enable:
            flag_jog_enable = False
            self.ids.bt_jog_enable.md_bg_color = "#196BA5"
        else:
            flag_jog_enable = True
            self.ids.bt_jog_enable.md_bg_color = "#ee2222"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3073, flag_jog_enable, slave=1) #M1
                modbus_client.close()
        except:
            toast("error send flag_jog_enable data to PLC Slave")  

    def end_jog_cartesian(self):
        global flag_conn_stat
        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3083, False, slave=1) #M11
                modbus_client.write_coil(3084, False, slave=1) #M12
                modbus_client.write_coil(3093, False, slave=1) #M21
                modbus_client.write_coil(3094, False, slave=1) #M22
                modbus_client.write_coil(3103, False, slave=1) #M31
                modbus_client.write_coil(3104, False, slave=1) #M32
                modbus_client.close()
        except:
            toast("error send end_jog data to PLC Slave")  

    def exec_jog_x_p(self):
        global flag_conn_stat, flag_jog_req_x
        flag_jog_req_x = True
        self.ids.bt_jog_x_p.md_bg_color = "#ee2222"
        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3083, True, slave=1) #M11
                modbus_client.close()
        except:
            toast("error send exec_jog_x_p data to PLC Slave")  

    def exec_jog_x_n(self):
        global flag_conn_stat, flag_jog_req_x
        flag_jog_req_x = True
        self.ids.bt_jog_x_n.md_bg_color = "#ee2222"
        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3084, True, slave=1) #M12
                modbus_client.close()
        except:
            toast("error send exec_jog_x_n data to PLC Slave")     

    def end_jog_x(self):
        global flag_jog_req_x
        flag_jog_req_x = False
        self.ids.bt_jog_x_p.md_bg_color = "#196BA5"
        self.ids.bt_jog_x_n.md_bg_color = "#196BA5"
        self.end_jog_cartesian()

    def exec_jog_y_p(self):
        global flag_conn_stat, flag_jog_req_y
        flag_jog_req_y = True
        self.ids.bt_jog_y_p.md_bg_color = "#ee2222"
        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3093, True, slave=1) #M21
                modbus_client.close()
        except:
            toast("error send exec_jog_y_p data to PLC Slave")  

    def exec_jog_y_n(self):
        global flag_conn_stat, flag_jog_req_y
        flag_jog_req_y = True
        self.ids.bt_jog_y_n.md_bg_color = "#ee2222"
        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3094, True, slave=1) #M22
                modbus_client.close()
        except:
            toast("error send exec_jog_y_n data to PLC Slave")  

    def end_jog_y(self):
        global flag_jog_req_y
        flag_jog_req_y = False
        self.ids.bt_jog_y_p.md_bg_color = "#196BA5"
        self.ids.bt_jog_y_n.md_bg_color = "#196BA5"
        self.end_jog_cartesian()

    def exec_jog_z_p(self):
        global flag_conn_stat, flag_jog_req_z
        flag_jog_req_z = True
        self.ids.bt_jog_z_p.md_bg_color = "#ee2222"
        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3103, True, slave=1) #M31
                modbus_client.close()
        except:
            toast("error send exec_jog_z_p data to PLC Slave")  

    def exec_jog_z_n(self):
        global flag_conn_stat, flag_jog_req_z
        flag_jog_req_z = True
        self.ids.bt_jog_z_n.md_bg_color = "#ee2222"
        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3104, True, slave=1) #M32
                modbus_client.close()
        except:
            toast("error send exec_jog_z_n data to PLC Slave")  

    def end_jog_z(self):
        global flag_jog_req_z
        flag_jog_req_z = False
        self.ids.bt_jog_z_p.md_bg_color = "#196BA5"
        self.ids.bt_jog_z_n.md_bg_color = "#196BA5"
        self.end_jog_cartesian()

    def exec_operate_x(self):
        global flag_conn_stat, flag_operate_req_x
        global val_x_pos_sv
        global view_camera
        elev, azim, roll = view_camera
        
        flag_operate_req_x = True
        self.ids.bt_operate_x.md_bg_color = "#ee2222"
        val_x_pos_sv = float(self.ids.input_operate_x.text)
        self.update_graph(elev, azim, roll)

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3075, flag_operate_req_x, slave=1) #M13
                modbus_client.write_register(1522, self.sign_int(val_x_pos_sv), slave=1) #V1010
                modbus_client.close()
        except:
            toast("error send exec_operate_x and val_operate_x data to PLC Slave") 

    def end_operate_x(self):
        global flag_conn_stat, flag_operate_req_x
        flag_operate_req_x = False
        self.ids.bt_operate_x.md_bg_color = "#196BA5"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3075, flag_operate_req_x, slave=1) #M13
                modbus_client.close()
        except:
            toast("error send end_operate_x data to PLC Slave") 

    def exec_operate_y(self):
        global flag_conn_stat, flag_operate_req_y
        global val_y_pos_sv
        global view_camera
        elev, azim, roll = view_camera

        flag_operate_req_y = True
        self.ids.bt_operate_y.md_bg_color = "#ee2222"
        val_y_pos_sv = float(self.ids.input_operate_y.text)
        self.update_graph(elev, azim, roll)

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3085, flag_operate_req_y, slave=1) #M23
                modbus_client.write_register(1622, self.sign_int(val_y_pos_sv), slave=1) #V1110
                modbus_client.write_register(1622, self.sign_int(val_y_pos_sv), slave=1) #V1110
                modbus_client.close()
        except:
            toast("error send exec_operate_y and val_operate_y data to PLC Slave") 

    def end_operate_y(self):
        global flag_conn_stat, flag_operate_req_y
        flag_operate_req_y = False
        self.ids.bt_operate_y.md_bg_color = "#196BA5"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3085, flag_operate_req_y, slave=1) #M23
                modbus_client.close()
        except:
            toast("error send end_operate_y data to PLC Slave") 

    def exec_operate_z(self):
        global flag_conn_stat, flag_operate_req_z
        global val_z_pos_sv
        global view_camera
        elev, azim, roll = view_camera

        flag_operate_req_z = True
        self.ids.bt_operate_z.md_bg_color = "#ee2222"
        val_z_pos_sv = float(self.ids.input_operate_z.text)
        self.update_graph(elev, azim, roll)

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3101, flag_operate_req_z, slave=1) #M29
                modbus_client.write_register(3573, self.sign_int(val_z_pos_sv), slave=1) #V3061
                modbus_client.close()
        except:
            toast("error send exec_operate_z and val_operate_z data to PLC Slave")

    def end_operate_z(self):
        global flag_conn_stat, flag_operate_req_z
        flag_operate_req_z = False
        self.ids.bt_operate_z.md_bg_color = "#196BA5"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3101, flag_operate_req_z, slave=1) #M29
                modbus_client.close()
        except:
            toast("error send end_operate_z data to PLC Slave")

    def update_cartesian(self):
        global val_x_pos_sv, val_y_pos_sv, val_z_pos_sv
        global view_camera
        elev, azim, roll = view_camera
        
        try:
            val_x_pos_sv = float(self.ids.input_operate_x.text) if self.ids.input_operate_x.text != "" else 0
            val_y_pos_sv = float(self.ids.input_operate_y.text) if self.ids.input_operate_y.text != "" else 0
            val_z_pos_sv = float(self.ids.input_operate_z.text) if self.ids.input_operate_z.text != "" else 0
            self.update_graph(elev, azim, roll)
        except:
            toast("error supdate coordinate data")

    def end_jog_joint(self):
        global flag_conn_stat
        try:
            
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3083, False, slave=1) #M11
                modbus_client.write_coil(3084, False, slave=1) #M12
                modbus_client.write_coil(3093, False, slave=1) #M21
                modbus_client.write_coil(3094, False, slave=1) #M22
                modbus_client.write_coil(3103, False, slave=1) #M31
                modbus_client.write_coil(3104, False, slave=1) #M32
                modbus_client.close()
        except:
            toast("error send end_jog data to PLC Slave")  

    def exec_jog_j1_p(self):
        global flag_conn_stat, flag_jog_req_j1
        flag_jog_req_j1 = True
        self.ids.bt_jog_j1_p.md_bg_color = "#ee2222"
        try:
            self.update_joint_pos()
            self.update_joint_vel()            
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_register(1622, self.sign_int(16000), slave=1) #V1110
                modbus_client.write_coil(3083, True, slave=1) #M11
                modbus_client.close()
        except:
            toast("error send exec_jog_x_p data to PLC Slave")  

    def exec_jog_j1_n(self):
        global flag_conn_stat, flag_jog_req_j1
        flag_jog_req_j1 = True
        self.ids.bt_jog_j1_n.md_bg_color = "#ee2222"
        try:
            self.update_joint_pos()
            self.update_joint_vel()            
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_register(1622, self.sign_int(-16000), slave=1) #V1110
                modbus_client.write_coil(3084, True, slave=1) #M12
                modbus_client.close()
        except:
            toast("error send exec_jog_x_n data to PLC Slave")     

    def end_jog_j1(self):
        global flag_jog_req_j1
        flag_jog_req_j1 = False
        self.ids.bt_jog_j1_p.md_bg_color = "#196BA5"
        self.ids.bt_jog_j1_n.md_bg_color = "#196BA5"
        self.end_jog_joint()

    def exec_jog_j2_p(self):
        global flag_conn_stat, flag_jog_req_j2
        flag_jog_req_j2 = True
        self.ids.bt_jog_j2_p.md_bg_color = "#ee2222"
        try:
            self.update_joint_pos()
            self.update_joint_vel()            
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_register(1722, self.sign_int(16000), slave=1) #V1210
                modbus_client.write_coil(3093, True, slave=1) #M21
                modbus_client.close()
        except:
            toast("error send exec_jog_y_p data to PLC Slave")  

    def exec_jog_j2_n(self):
        global flag_conn_stat, flag_jog_req_j2
        flag_jog_req_j2 = True
        self.ids.bt_jog_j2_n.md_bg_color = "#ee2222"
        try:
            self.update_joint_pos()
            self.update_joint_vel()            
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_register(1722, self.sign_int(-16000), slave=1) #V1110
                modbus_client.write_coil(3094, True, slave=1) #M22
                modbus_client.close()
        except:
            toast("error send exec_jog_y_n data to PLC Slave")  

    def end_jog_j2(self):
        global flag_jog_req_j2
        flag_jog_req_j2 = False
        self.ids.bt_jog_j2_p.md_bg_color = "#196BA5"
        self.ids.bt_jog_j2_n.md_bg_color = "#196BA5"
        self.end_jog_joint()

    def exec_jog_j3_p(self):
        global flag_conn_stat, flag_jog_req_j3
        flag_jog_req_j3 = True
        self.ids.bt_jog_j3_p.md_bg_color = "#ee2222"
        try:
            self.update_joint_pos()
            self.update_joint_vel()            
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_register(1822, self.sign_int(16000), slave=1) #V1310
                modbus_client.write_coil(3103, True, slave=1) #M31
                modbus_client.close()
        except:
            toast("error send exec_jog_z_p data to PLC Slave")  

    def exec_jog_j3_n(self):
        global flag_conn_stat, flag_jog_req_j3
        flag_jog_req_j3 = True
        self.ids.bt_jog_j3_n.md_bg_color = "#ee2222"
        try:
            self.update_joint_pos()
            self.update_joint_vel()            
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_register(1822, self.sign_int(-16000), slave=1) #V1310
                modbus_client.write_coil(3104, True, slave=1) #M32
                modbus_client.close()
        except:
            toast("error send exec_jog_z_n data to PLC Slave")  

    def end_jog_j3(self):
        global flag_jog_req_j3
        flag_jog_req_j3 = False
        self.ids.bt_jog_j3_p.md_bg_color = "#196BA5"
        self.ids.bt_jog_j3_n.md_bg_color = "#196BA5"
        self.end_jog_joint()

    def exec_operate_j1(self):
        global flag_conn_stat, flag_operate_req_j1
        global val_j1_pos_sv, val_j1_vel
        global view_camera
        elev, azim, roll = view_camera
        
        flag_operate_req_j1 = True
        self.ids.bt_operate_j1.md_bg_color = "#ee2222"
        val_j1_pos_sv = float(self.ids.input_operate_j1.text)
        self.update_graph(elev, azim, roll)

        try:
            self.update_joint_pos()
            self.update_joint_vel()            
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3085, flag_operate_req_j1, slave=1) #M13
                modbus_client.close()
        except:
            toast("error send exec_operate_j1 and val_operate_j1 data to PLC Slave") 

    def end_operate_j1(self):
        global flag_conn_stat, flag_operate_req_j1
        flag_operate_req_j1 = False
        self.ids.bt_operate_j1.md_bg_color = "#196BA5"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3085, flag_operate_req_j1, slave=1) #M13
                modbus_client.close()
        except:
            toast("error send end_operate_j1 data to PLC Slave") 

    def exec_operate_j2(self):
        global flag_conn_stat, flag_operate_req_j2
        global val_j2_pos_sv, val_j2_vel
        global view_camera
        elev, azim, roll = view_camera

        flag_operate_req_j2 = True
        self.ids.bt_operate_j2.md_bg_color = "#ee2222"
        val_j2_pos_sv = float(self.ids.input_operate_j2.text)
        self.update_graph(elev, azim, roll)

        try:
            self.update_joint_pos()
            self.update_joint_vel()            
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3095, flag_operate_req_j2, slave=1) #M23
                modbus_client.close()
        except:
            toast("error send exec_operate_j2 and val_operate_j2 data to PLC Slave") 

    def end_operate_j2(self):
        global flag_conn_stat, flag_operate_req_j2
        flag_operate_req_j2 = False
        self.ids.bt_operate_j2.md_bg_color = "#196BA5"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3095, flag_operate_req_j2, slave=1) #M23
                modbus_client.close()
        except:
            toast("error send end_operate_j2 data to PLC Slave") 

    def exec_operate_j3(self):
        global flag_conn_stat, flag_operate_req_j3
        global val_j3_pos_sv, val_j3_vel
        global view_camera
        elev, azim, roll = view_camera

        flag_operate_req_j3 = True
        self.ids.bt_operate_j3.md_bg_color = "#ee2222"
        val_j3_pos_sv = float(self.ids.input_operate_j3.text)
        self.update_graph(elev, azim, roll)

        try:
            self.update_joint_pos()
            self.update_joint_vel()
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3105, flag_operate_req_j3, slave=1) #M33
                modbus_client.close()
        except:
            toast("error send exec_operate_j3 and val_operate_j3 data to PLC Slave")

    def end_operate_j3(self):
        global flag_conn_stat, flag_operate_req_j3
        flag_operate_req_j3 = False
        self.ids.bt_operate_j3.md_bg_color = "#196BA5"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3105, flag_operate_req_j3, slave=1) #M33
                modbus_client.close()
        except:
            toast("error send end_operate_j3 data to PLC Slave")

    def update_joint_pos(self):
        global val_j1_pos_sv, val_j2_pos_sv, val_j3_pos_sv
        global val_j1_pos_pv, val_j2_pos_pv, val_j3_pos_pv
        global val_x_pos_pv, val_y_pos_pv, val_z_pos_pv
        global view_camera
        elev, azim, roll = view_camera
        
        try:
            val_j1_pos_sv = float(self.ids.input_operate_j1.text) if self.ids.input_operate_j1.text != "" else 0
            val_j2_pos_sv = float(self.ids.input_operate_j2.text) if self.ids.input_operate_j2.text != "" else 0
            val_j3_pos_sv = float(self.ids.input_operate_j3.text) if self.ids.input_operate_j3.text != "" else 0

            val_j1_pulse = (val_j1_pos_sv - val_j1_pos_pv) * PULSE_PER_REV * GEAR_RATIO / 360
            val_j2_pulse = (val_j2_pos_sv - val_j2_pos_pv) * PULSE_PER_REV * GEAR_RATIO / 360
            val_j3_pulse = (val_j3_pos_sv - val_j3_pos_pv) * PULSE_PER_REV * GEAR_RATIO / 360

            bot = SimulatedDeltaBot(LINK_JOINT_LENGTH, LINK_PARALLEL_LENGTH,
                                    JOINT_DISPLACEMENT, EFFECTOR_DISPLACEMENT)
            servo_angle = np.array([val_j1_pos_pv, val_j2_pos_pv, val_j3_pos_pv], dtype=int)

            fk_result = bot.forward(*servo_angle)
            
            val_x_pos_pv = fk_result[0]
            val_y_pos_pv = fk_result[1]
            val_z_pos_pv = fk_result[2]

            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_register(1622, self.sign_int(val_j1_pulse), slave=1) #V1110
                modbus_client.write_register(1722, self.sign_int(val_j2_pulse), slave=1) #V1210
                modbus_client.write_register(1822, self.sign_int(val_j3_pulse), slave=1) #V1310
                modbus_client.close()

            self.update_graph(elev, azim, roll)
        except:
            toast("error update joint angle data")

    def update_joint_vel(self):
        global val_j1_vel, val_j2_vel, val_j3_vel
        global view_camera
        elev, azim, roll = view_camera
        
        try:
            val_j1_vel = float(self.ids.input_vel_j1.text) if self.ids.input_vel_j1.text != "" else 0
            val_j2_vel = float(self.ids.input_vel_j2.text) if self.ids.input_vel_j2.text != "" else 0
            val_j3_vel = float(self.ids.input_vel_j3.text) if self.ids.input_vel_j3.text != "" else 0

            val_j1_pvel = val_j1_vel * PULSE_PER_REV * GEAR_RATIO / 360 
            val_j2_pvel = val_j1_vel * PULSE_PER_REV * GEAR_RATIO / 360
            val_j3_pvel = val_j1_vel * PULSE_PER_REV * GEAR_RATIO / 360
            
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_register(1612, self.sign_int(val_j1_pvel), slave=1) #V1100
                modbus_client.write_register(1712, self.sign_int(val_j2_pvel), slave=1) #V1200
                modbus_client.write_register(1812, self.sign_int(val_j3_pvel), slave=1) #V1300
                modbus_client.close()

            self.update_graph(elev, azim, roll)         
        except:
            toast("error supdate joint velocity data")

    def exec_origin(self):
        global flag_conn_stat, flag_origin_req
        global val_x_pos_sv, val_y_pos_sv, val_z_pos_sv
        global view_camera
        elev, azim, roll = view_camera

        flag_origin_req = True
        self.ids.bt_origin.md_bg_color = "#ee2222"

        try:
            val_x_pos_sv = 0.
            val_y_pos_sv = 0.
            val_z_pos_sv = -200.

            self.ids.input_operate_x.text = str(val_x_pos_sv)
            self.ids.input_operate_y.text = str(val_y_pos_sv)
            self.ids.input_operate_z.text = str(val_z_pos_sv)

            self.update_graph(elev, azim, roll)

            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3074, flag_origin_req, slave=1) #M2
                modbus_client.close()
        except:
            toast("error send flag_origin_req data to PLC Slave")

    def end_origin(self):
        global flag_conn_stat, flag_origin_req
        flag_origin_req = False
        self.ids.bt_origin.md_bg_color = "#196BA5"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3074, flag_origin_req, slave=1) #M2
                modbus_client.close()
        except:
            toast("error send flag_origin_req data to PLC Slave")

    def exec_reset(self):
        global flag_conn_stat, flag_reset
        global val_x_pos_sv, val_y_pos_sv, val_z_pos_sv
        global view_camera
        elev, azim, roll = view_camera
        
        flag_reset = True
        self.ids.bt_reset.md_bg_color = "#ee2222"

        try:
            val_x_pos_sv = 0.
            val_y_pos_sv = 0.
            val_z_pos_sv = -200.

            self.ids.input_operate_x.text = str(val_x_pos_sv)
            self.ids.input_operate_y.text = str(val_y_pos_sv)
            self.ids.input_operate_z.text = str(val_z_pos_sv)

            self.update_graph(elev, azim, roll)

            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3075, flag_reset, slave=1) #M3
                modbus_client.close()
        except:
            toast("error send flag_reset data to PLC Slave")

    def end_reset(self):
        global flag_conn_stat, flag_reset
        flag_reset = False
        self.ids.bt_reset.md_bg_color = "#196BA5"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3075, flag_reset, slave=1) #M3
                modbus_client.close()
        except:
            toast("error send flag_reset data to PLC Slave")

    def screen_main_menu(self):
        self.screen_manager.current = 'screen_main_menu'

    def screen_pipe_setting(self):
        self.screen_manager.current = 'screen_pipe_setting'

    def screen_machine_setting(self):
        self.screen_manager.current = 'screen_machine_setting'

    def screen_advanced_setting(self):
        self.screen_manager.current = 'screen_advanced_setting'

    def screen_operate_manual(self):
        self.screen_manager.current = 'screen_operate_manual'

    def screen_operate_auto(self):
        self.screen_manager.current = 'screen_operate_auto'

    def screen_compile(self):
        self.screen_manager.current = 'screen_compile'

    def exec_shutdown(self):
        os.system("shutdown /s /t 1") #for windows os
        toast("shutting down system")
        # os.system("shutdown -h 1")

class ScreenOperateAuto(MDScreen):
    def __init__(self, **kwargs):       
        super(ScreenOperateAuto, self).__init__(**kwargs)
        self.file_manager = MDFileManager(exit_manager=self.exit_manager, select_path=self.select_path)
        Clock.schedule_once(self.delayed_init, 5)

    def delayed_init(self, dt):
        self.reload()

    def update_view(self, direction):
        global view_camera

        elev, azim, roll = view_camera
        
        if(direction == 0):
            print(elev)
            elev += 20

        if(direction == 1):
            print(elev)
            elev -= 20
        
        if(direction == 2):
            azim += 20
        
        if(direction == 3):
            azim -= 20
        
        view_camera = np.array([elev, azim, roll])        
        self.update_graph(elev, azim, roll)

    def reload(self):
        global data_base_process
        
        self.update_graph()
        self.send_data()

    def file_manager_open(self):
        self.file_manager.show(os.path.expanduser(os.getcwd() + "\data"))  # output manager to the screen
        self.manager_open = True

    def select_path(self, path: str):
        try:
            self.exit_manager(path)
        except:
            toast("error select file path")

    def exit_manager(self, *args):
        global data_base_process, data_base_config
        try:
            data_set = np.loadtxt(*args, delimiter="\t", encoding=None, skiprows=1)
            data_base_load = data_set.T
            data_base_process = data_base_load[:3,:]
            data_base_config = data_base_load[3:,:]
            self.reload()

            self.manager_open = False
            self.file_manager.close()
        except:
            toast("error open file")
            self.file_manager.close()
    
    def send_data(self):
        global val_x_step, val_y_step, val_z_step

        global data_base_process
        global data_base_config
        global val_machine_die_radius

        global conf_x_speed_step, conf_y_speed_step, conf_z_speed_step
        global conf_bed_pos_step 

        val_x_step = data_base_process[0,:]
        val_y_step = data_base_process[1,:] 
        val_z_step = data_base_process[2,:] 

        conf_x_speed_step = data_base_config[0,:]
        conf_y_speed_step = data_base_config[1,:]
        conf_z_speed_step = data_base_config[2,:]
        conf_bed_pos_step = data_base_config[3,:]
        print(data_base_config)

        val_x_absolute_step = np.zeros(10)
        val_y_linear_absolute_step = np.zeros(10)
        # y linear offset = 2 pi * r * die radius / 360 
        # (conversion from ying movement to x offset linear movement)
        val_y_linear_offset_step = val_machine_die_radius * 2 * np.pi * val_y_step / 360
        # val_y_linear_offset_step = val_machine_die_radius * val_y_step / 360

        # setting val_advanced_receive_pos_x as first cycle position set value x
        val_x_absolute_step[0] = int(val_x_step[0] + val_advanced_receive_pos_x)
        val_y_linear_absolute_step[0] = int(val_x_absolute_step[0] + val_y_linear_offset_step[0])        

        for i in range(1,10):
            # x absolute = x offset + last x absolute + y linear offset
            val_x_absolute_step[i] = int(val_x_absolute_step[i-1] + val_x_step[i])
            
            if val_x_absolute_step[i] > val_machine_eff_length:
                val_x_absolute_step[i] = int(val_x_step[i] + val_advanced_receive_pos_x)

        for i in range(1,9):
            val_y_linear_absolute_step[i] = int(val_x_absolute_step[i] + val_y_linear_offset_step[i])

        val_z_absolute_step = np.zeros(10)
        val_z_absolute_step[0] = val_z_step[0]
        for i in range(1,10):
            # z absolute = z offset + last z absolute
            val_z_absolute_step[i] = int(val_z_step[i] + val_z_absolute_step[i-1])

        list_val_x_absolute_step = val_x_absolute_step.astype(int).tolist()
        list_val_y_step = val_y_step.astype(int).tolist()
        list_val_z_absolute_step = val_z_absolute_step.astype(int).tolist()
        # list_val_z_step = val_z_step.astype(int).tolist()
        list_val_y_linear_absolute_step = val_y_linear_absolute_step.astype(int).tolist()

        list_conf_x_speed_step = conf_x_speed_step.astype(int).tolist()
        list_conf_y_speed_step = conf_y_speed_step.astype(int).tolist()
        list_conf_z_speed_step = conf_z_speed_step.astype(int).tolist()
        list_conf_bed_pos_step = conf_bed_pos_step.astype(bool).tolist()

        print("list_val_x_absolute_step", list_val_x_absolute_step)
        print("list_val_y_step", list_val_y_step)
        print("list_val_z_absolute_step", list_val_z_absolute_step)
        print("list_val_y_linear_absolute_step", list_val_y_linear_absolute_step)

        try:
            if flag_conn_stat:
                modbus_client.connect()
                # modbus_client.write_register(3523, int(val_x_step[0]), slave=1) #V3011
                # modbus_client.write_register(3553, int(val_y_step[0]), slave=1) #V3011
                # modbus_client.write_register(3583, int(val_z_step[0]), slave=1) #V3011

                # modbus_client.write_register(3524, int(val_x_step[1]), slave=1) #V3011
                # modbus_client.write_register(3554, int(val_y_step[1]), slave=1) #V3011
                # modbus_client.write_register(3584, int(val_z_step[1]), slave=1) #V3011

                modbus_client.write_registers(3523, list_val_x_absolute_step, slave=1) #V3011
                modbus_client.write_registers(3553, list_val_y_step, slave=1) #V3041
                modbus_client.write_registers(3583, list_val_z_absolute_step, slave=1) #V3071
                # modbus_client.write_registers(3583, list_val_z_step, slave=1) #V3071
                modbus_client.write_registers(3623, list_val_y_linear_absolute_step, slave=1) #V3111

                modbus_client.write_registers(3723, list_conf_x_speed_step, slave=1) #V3211
                modbus_client.write_registers(3753, list_conf_y_speed_step, slave=1) #V3241
                modbus_client.write_registers(3783, list_conf_z_speed_step, slave=1) #V3271
                modbus_client.write_coils(3383, list_conf_bed_pos_step, slave=1) #M311
                # modbus_client.write_coils(3093, [False, False, False, False, False, False], slave=1) #M21 - M26
                modbus_client.close()
        except Exception as e:
            toast(e) 
            
    def update_graph(self, elev=45, azim=60, roll=0):
        global val_pipe_length
        global val_pipe_diameter
        global val_pipe_thickness

        global val_x_step
        global val_y_step
        global val_z_step

        global data_base_process
        view_camera = elev, azim, roll
        try:
            val_x_step = data_base_process[0,:]
            val_y_step = data_base_process[1,:] 
            val_z_step = data_base_process[2,:] 

            self.ids.pipe_bended_illustration.clear_widgets()

            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.fig.set_facecolor("#eeeeee")
            # self.fig.tight_layout()

            offset_length = val_x_step
            y_angle = val_y_step / 180 * np.pi
            z_angle = val_z_step / 180 * np.pi
            pipe_radius = val_pipe_diameter / 2

            Uo = np.linspace(0, 2 * np.pi, 30)
            Yo = np.linspace(0, 0, 5)
            Uo, Yo = np.meshgrid(Uo, Yo)
            Xo = pipe_radius * np.cos(Uo) - val_machine_die_radius
            Zo = pipe_radius * np.sin(Uo)
            
            X0, Y0, Z0 = self.simulate(Xo, Yo, Zo, offset_length[0], y_angle[0], z_angle[0])
            X1, Y1, Z1 = self.simulate(X0, Y0, Z0, offset_length[1], y_angle[1], z_angle[1])
            X2, Y2, Z2 = self.simulate(X1, Y1, Z1, offset_length[2], y_angle[2], z_angle[2])
            X3, Y3, Z3 = self.simulate(X2, Y2, Z2, offset_length[3], y_angle[3], z_angle[3])
            X4, Y4, Z4 = self.simulate(X3, Y3, Z3, offset_length[4], y_angle[4], z_angle[4])
            X5, Y5, Z5 = self.simulate(X4, Y4, Z4, offset_length[5], y_angle[5], z_angle[5])
            X6, Y6, Z6 = self.simulate(X5, Y5, Z5, offset_length[6], y_angle[6], z_angle[6])
            X7, Y7, Z7 = self.simulate(X6, Y6, Z6, offset_length[7], y_angle[7], z_angle[7])
            X8, Y8, Z8 = self.simulate(X7, Y7, Z7, offset_length[8], y_angle[8], z_angle[8])
            X9, Y9, Z9 = self.simulate(X8, Y8, Z8, offset_length[9], y_angle[9], z_angle[9])

            self.ax.plot_surface(X9, Y9, Z9, color='gray', alpha=1)
            # self.ax.set_box_aspect(aspect=(1, 1, 1))
            self.ax.set_aspect('equal')
            # self.ax.set_xlim([0, 6000])
            # self.ax.set_ylim([-100, 100])
            # self.ax.set_zlim([-100, 100])
            # self.ax.axis('off')
            self.ax.view_init(elev=view_camera[0], azim=view_camera[1], roll=view_camera[2])
            self.ids.pipe_bended_illustration.add_widget(FigureCanvasKivyAgg(self.fig))   
        except:
            toast("error update pipe ying process illustration")
   
    def simulate(self, prev_X, prev_Y, prev_Z, offset_length, y_angle, z_angle):
        global flag_run
        global val_x_step
        global val_y_step
        global val_z_step

        global val_pipe_diameter
        global val_machine_die_radius

        pipe_radius = val_pipe_diameter / 2
        # step 1 : create straight pipe
        # straight pipe
        Ua = np.linspace(0, 2 * np.pi, 30)
        Ya = np.linspace(offset_length, 0, 5)
        Ua, Ya = np.meshgrid(Ua, Ya)
        Xa = pipe_radius * np.cos(Ua) - val_machine_die_radius
        Za = pipe_radius * np.sin(Ua)
        # combine become one object with previous mesh
        Xa = np.append(prev_X, Xa, axis=0)
        Ya = np.append(prev_Y + offset_length, Ya, axis=0)
        Za = np.append(prev_Z, Za, axis=0)

        # step 2 : create yed pipe
        # theta: poloidal angle; phi: toroidal angle 
        theta = np.linspace(0, 2 * np.pi, 30) 
        phi   = np.linspace(0, y_angle, 30) 
        theta, phi = np.meshgrid(theta, phi) 
        # torus parametrization 
        Xb = (val_machine_die_radius + pipe_radius * np.cos(theta)) * -np.cos(phi)
        Yb = (val_machine_die_radius + pipe_radius * np.cos(theta)) * -np.sin(phi)
        Zb = pipe_radius * np.sin(theta) 

        # step 3 : combine become one object
        Xc = np.append(Xa, Xb, axis=0)
        Yc = np.append(Ya, Yb, axis=0)
        Zc = np.append(Za, Zb, axis=0)

        # step 4 : rotate  object at Z axis (C axis)
        Xd = np.cos(y_angle) * Xc + np.sin(y_angle) * Yc
        Yd = -np.sin(y_angle) * Xc + np.cos(y_angle) * Yc
        Zd = Zc

        # step 5 : translate to origin, rotate  object at Y axis (B axis), translate back to previous position
        # translate
        Xe = Xd + val_machine_die_radius
        Ze = Zd
        # rotate
        Xf = np.cos(z_angle) * Xe + -np.sin(z_angle) * Ze
        Zf = np.sin(z_angle) * Xe + np.cos(z_angle) * Ze
        # translate back
        Xf = Xf - val_machine_die_radius
        Yf = Yd

        return Xf, Yf, Zf
    
    def exec_mode(self):
        global flag_conn_stat, flag_mode

        if flag_mode:
            flag_mode = False
        else:
            flag_mode = True

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3072, flag_mode, slave=1) #M0
                modbus_client.close()
        except Exception as e:
            toast(e) 

    def exec_start(self):
        global flag_conn_stat, flag_run
        flag_run = True
        self.ids.bt_start.md_bg_color = "#ee2222"
        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3073, flag_run, slave=1) #M1
                modbus_client.close()
        except:
            toast("error send flag_run data to PLC Slave") 

    def end_start(self):
        self.ids.bt_start.md_bg_color = "#196BA5"


    def exec_stop(self):
        global flag_conn_stat, flag_run
        flag_run = False
        self.ids.bt_stop.md_bg_color = "#ee2222"
        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3073, flag_run, slave=1) #M1
                modbus_client.close()
        except:
            toast("error send flag_run data to PLC Slave") 

    def end_stop(self):
        self.ids.bt_stop.md_bg_color = "#196BA5"

    def exec_origin(self):
        global flag_conn_stat, flag_origin_req
        flag_origin_req = True
        self.ids.bt_origin.md_bg_color = "#ee2222"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3102, flag_origin_req, slave=1) #M30
                modbus_client.close()
        except:
            toast("error send flag_origin_req data to PLC Slave")

    def end_origin(self):
        global flag_conn_stat, flag_origin_req
        flag_origin_req = False
        self.ids.bt_origin.md_bg_color = "#196BA5"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3102, flag_origin_req, slave=1) #M30
                modbus_client.close()
        except:
            toast("error send flag_origin_req data to PLC Slave")

    def exec_reset(self):
        global flag_conn_stat, flag_reset
        flag_reset = True
        self.ids.bt_reset.md_bg_color = "#ee2222"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3075, flag_reset, slave=1) #M3
                modbus_client.close()
        except:
            toast("error send flag_reset data to PLC Slave")

    def end_reset(self):
        global flag_conn_stat, flag_reset
        flag_reset = False
        self.ids.bt_reset.md_bg_color = "#196BA5"

        try:
            if flag_conn_stat:
                modbus_client.connect()
                modbus_client.write_coil(3075, flag_reset, slave=1) #M3
                modbus_client.close()
        except:
            toast("error send flag_reset data to PLC Slave")
    
    def screen_main_menu(self):
        self.screen_manager.current = 'screen_main_menu'

    def screen_pipe_setting(self):
        self.screen_manager.current = 'screen_pipe_setting'

    def screen_machine_setting(self):
        self.screen_manager.current = 'screen_machine_setting'

    def screen_advanced_setting(self):
        self.screen_manager.current = 'screen_advanced_setting'

    def screen_operate_manual(self):
        self.screen_manager.current = 'screen_operate_manual'

    def screen_operate_auto(self):
        self.screen_manager.current = 'screen_operate_auto'

    def screen_compile(self):
        self.screen_manager.current = 'screen_compile'

    def exec_shutdown(self):
        os.system("shutdown /s /t 1") #for windows os
        toast("shutting down system")
        # os.system("shutdown -h 1")

class ScreenCompile(MDScreen):
    def __init__(self, **kwargs):
        global data_base_config
        global conf_x_speed_step, conf_y_speed_step, conf_z_speed_step, conf_bed_pos_step

        super(ScreenCompile, self).__init__(**kwargs)
        self.file_manager = MDFileManager(exit_manager=self.exit_manager, select_path=self.select_path)
        for i in range(0,9):
            data_base_config[0,i] = conf_x_speed_step[i]
            data_base_config[1,i] = conf_y_speed_step[i]
            data_base_config[2,i] = conf_z_speed_step[i]
            data_base_config[3,i] = conf_bed_pos_step[i]

    def file_manager_open(self):
        self.file_manager.show(os.path.expanduser(os.getcwd() + "\data"))  # output manager to the screen
        self.manager_open = True

    def select_path(self, path: str):
        try:
            path_name = os.path.expanduser(os.getcwd() + "\data\\")
            filename = path.replace(path_name, "")
            filename = filename.replace(".gcode", "")
            self.ids.input_file_name.text = filename
            self.exit_manager(path)
        except:
            toast("error select file path")

    def exit_manager(self, *args):
        global data_base_process, data_base_config
        '''Called when the user reaches the root of the directory tree.'''
        try:
            data_set = np.loadtxt(*args, delimiter="\t", encoding=None, skiprows=1)
            data_base_load = data_set.T
            data_base_process = data_base_load[:3,:]
            data_base_config = data_base_load[3:,:]

            self.update_text_data()
            self.update_text_config()
            self.update_graph()

            self.manager_open = False
            self.file_manager.close()
        except Exception as e:
            toast("error open file")
            print(e)
            self.file_manager.close()

    def update_text_data(self):
        global flag_conn_stat
        global val_pipe_length, val_pipe_diameter, val_pipe_thickness
        global val_x_step, val_y_step, val_z_step
        global data_base_process

        val_x_step = data_base_process[0,:]
        val_y_step = data_base_process[1,:] 
        val_z_step = data_base_process[2,:] 
    
        self.ids.input_x_step0.text = str(val_x_step[0])
        self.ids.input_y_step0.text = str(val_y_step[0])
        self.ids.input_z_step0.text = str(val_z_step[0])

        self.ids.input_x_step1.text = str(val_x_step[1])
        self.ids.input_y_step1.text = str(val_y_step[1])
        self.ids.input_z_step1.text = str(val_z_step[1])

        self.ids.input_x_step2.text = str(val_x_step[2])
        self.ids.input_y_step2.text = str(val_y_step[2])
        self.ids.input_z_step2.text = str(val_z_step[2])

        self.ids.input_x_step3.text = str(val_x_step[3])
        self.ids.input_y_step3.text = str(val_y_step[3])
        self.ids.input_z_step3.text = str(val_z_step[3])

        self.ids.input_x_step4.text = str(val_x_step[4])
        self.ids.input_y_step4.text = str(val_y_step[4])
        self.ids.input_z_step4.text = str(val_z_step[4])

        self.ids.input_x_step5.text = str(val_x_step[5])
        self.ids.input_y_step5.text = str(val_y_step[5])
        self.ids.input_z_step5.text = str(val_z_step[5])

        self.ids.input_x_step6.text = str(val_x_step[6])
        self.ids.input_y_step6.text = str(val_y_step[6])
        self.ids.input_z_step6.text = str(val_z_step[6])

        self.ids.input_x_step7.text = str(val_x_step[7])
        self.ids.input_y_step7.text = str(val_y_step[7])
        self.ids.input_z_step7.text = str(val_z_step[7])

        self.ids.input_x_step8.text = str(val_x_step[8])
        self.ids.input_y_step8.text = str(val_y_step[8])
        self.ids.input_z_step8.text = str(val_z_step[8])

        self.ids.input_x_step9.text = str(val_x_step[9])
        self.ids.input_y_step9.text = str(val_y_step[9])
        self.ids.input_z_step9.text = str(val_z_step[9]) 

    def update_text_config(self):
        global flag_conn_stat
        global conf_x_speed_step, conf_y_speed_step, conf_z_speed_step, conf_bed_pos_step
        global data_base_config

        conf_x_speed_step = data_base_config[0,:]
        conf_y_speed_step = data_base_config[1,:] 
        conf_z_speed_step = data_base_config[2,:] 
        conf_bed_pos_step = data_base_config[3,:] 
    
        self.ids.bt_x_speed_step0.text = str(int(conf_x_speed_step[0]))
        self.ids.bt_y_speed_step0.text = str(int(conf_y_speed_step[0]))
        self.ids.bt_z_speed_step0.text = str(int(conf_z_speed_step[0]))
        self.ids.bt_bed_pos0.text = "UP" if conf_bed_pos_step[0] == 1 else "DN"

        self.ids.bt_x_speed_step1.text = str(int(conf_x_speed_step[1]))
        self.ids.bt_y_speed_step1.text = str(int(conf_y_speed_step[1]))
        self.ids.bt_z_speed_step1.text = str(int(conf_z_speed_step[1]))
        self.ids.bt_bed_pos1.text = "UP" if conf_bed_pos_step[1] == 1 else "DN"

        self.ids.bt_x_speed_step2.text = str(int(conf_x_speed_step[2]))
        self.ids.bt_y_speed_step2.text = str(int(conf_y_speed_step[2]))
        self.ids.bt_z_speed_step2.text = str(int(conf_z_speed_step[2]))
        self.ids.bt_bed_pos2.text = "UP" if conf_bed_pos_step[2] == 1 else "DN"

        self.ids.bt_x_speed_step3.text = str(int(conf_x_speed_step[3]))
        self.ids.bt_y_speed_step3.text = str(int(conf_y_speed_step[3]))
        self.ids.bt_z_speed_step3.text = str(int(conf_z_speed_step[3]))
        self.ids.bt_bed_pos3.text = "UP" if conf_bed_pos_step[3] == 1 else "DN"

        self.ids.bt_x_speed_step4.text = str(int(conf_x_speed_step[4]))
        self.ids.bt_y_speed_step4.text = str(int(conf_y_speed_step[4]))
        self.ids.bt_z_speed_step4.text = str(int(conf_z_speed_step[4]))
        self.ids.bt_bed_pos4.text = "UP" if conf_bed_pos_step[4] == 1 else "DN"

        self.ids.bt_x_speed_step5.text = str(int(conf_x_speed_step[5]))
        self.ids.bt_y_speed_step5.text = str(int(conf_y_speed_step[5]))
        self.ids.bt_z_speed_step5.text = str(int(conf_z_speed_step[5]))
        self.ids.bt_bed_pos5.text = "UP" if conf_bed_pos_step[5] == 1 else "DN"

        self.ids.bt_x_speed_step6.text = str(int(conf_x_speed_step[6]))
        self.ids.bt_y_speed_step6.text = str(int(conf_y_speed_step[6]))
        self.ids.bt_z_speed_step6.text = str(int(conf_z_speed_step[6]))
        self.ids.bt_bed_pos6.text = "UP" if conf_bed_pos_step[6] == 1 else "DN"

        self.ids.bt_x_speed_step7.text = str(int(conf_x_speed_step[7]))
        self.ids.bt_y_speed_step7.text = str(int(conf_y_speed_step[7]))
        self.ids.bt_z_speed_step7.text = str(int(conf_z_speed_step[7]))
        self.ids.bt_bed_pos7.text = "UP" if conf_bed_pos_step[7] == 1 else "DN"

        self.ids.bt_x_speed_step8.text = str(int(conf_x_speed_step[8]))
        self.ids.bt_y_speed_step8.text = str(int(conf_y_speed_step[8]))
        self.ids.bt_z_speed_step8.text = str(int(conf_z_speed_step[8]))
        self.ids.bt_bed_pos8.text = "UP" if conf_bed_pos_step[8] == 1 else "DN"

        self.ids.bt_x_speed_step9.text = str(int(conf_x_speed_step[9]))
        self.ids.bt_y_speed_step9.text = str(int(conf_y_speed_step[9]))
        self.ids.bt_z_speed_step9.text = str(int(conf_z_speed_step[9]))
        self.ids.bt_bed_pos9.text = "UP" if conf_bed_pos_step[9] == 1 else "DN"

    def update_view(self, direction):
        global view_camera

        elev, azim, roll = view_camera
        
        if(direction == 0):
            print(elev)
            elev += 20

        if(direction == 1):
            print(elev)
            elev -= 20
        
        if(direction == 2):
            azim += 20
        
        if(direction == 3):
            azim -= 20
        
        view_camera = np.array([elev, azim, roll])        
        self.update_graph(elev, azim, roll)
    
    def update(self):
        val_x_step[0] = float(self.ids.input_x_step0.text)
        val_y_step[0] = float(self.ids.input_y_step0.text)
        val_z_step[0] = float(self.ids.input_z_step0.text)

        val_x_step[1] = float(self.ids.input_x_step1.text)
        val_y_step[1] = float(self.ids.input_y_step1.text)
        val_z_step[1] = float(self.ids.input_z_step1.text)

        val_x_step[2] = float(self.ids.input_x_step2.text)
        val_y_step[2] = float(self.ids.input_y_step2.text)
        val_z_step[2] = float(self.ids.input_z_step2.text)

        val_x_step[3] = float(self.ids.input_x_step3.text)
        val_y_step[3] = float(self.ids.input_y_step3.text)
        val_z_step[3] = float(self.ids.input_z_step3.text)

        val_x_step[4] = float(self.ids.input_x_step4.text)
        val_y_step[4] = float(self.ids.input_y_step4.text)
        val_z_step[4] = float(self.ids.input_z_step4.text)

        val_x_step[5] = float(self.ids.input_x_step5.text)
        val_y_step[5] = float(self.ids.input_y_step5.text)
        val_z_step[5] = float(self.ids.input_z_step5.text)

        val_x_step[6] = float(self.ids.input_x_step6.text)
        val_y_step[6] = float(self.ids.input_y_step6.text)
        val_z_step[6] = float(self.ids.input_z_step6.text)

        val_x_step[7] = float(self.ids.input_x_step7.text)
        val_y_step[7] = float(self.ids.input_y_step7.text)
        val_z_step[7] = float(self.ids.input_z_step7.text)

        val_x_step[8] = float(self.ids.input_x_step8.text)
        val_y_step[8] = float(self.ids.input_y_step8.text)
        val_z_step[8] = float(self.ids.input_z_step8.text)

        val_x_step[9] = float(self.ids.input_x_step9.text)
        val_y_step[9] = float(self.ids.input_y_step9.text)
        val_z_step[9] = float(self.ids.input_z_step9.text)
   
    def update_config(self):
        conf_x_speed_step[0] = int(self.ids.input_x_speed_step0.text)
        conf_y_speed_step[0] = int(self.ids.input_y_speed_step0.text)
        conf_z_speed_step[0] = int(self.ids.input_z_speed_step0.text)
        conf_bed_pos_step[0] = 1 if self.ids.bt_bed_pos0.text == "UP" else 0

        conf_x_speed_step[1] = int(self.ids.input_x_speed_step1.text)
        conf_y_speed_step[1] = int(self.ids.input_y_speed_step1.text)
        conf_z_speed_step[1] = int(self.ids.input_z_speed_step1.text)
        conf_bed_pos_step[1] = 1 if self.ids.bt_bed_pos1.text == "UP" else 0

        conf_x_speed_step[2] = int(self.ids.input_x_speed_step2.text)
        conf_y_speed_step[2] = int(self.ids.input_y_speed_step2.text)
        conf_z_speed_step[2] = int(self.ids.input_z_speed_step2.text)
        conf_bed_pos_step[2] = 1 if self.ids.bt_bed_pos2.text == "UP" else 0

        conf_x_speed_step[3] = int(self.ids.input_x_speed_step3.text)
        conf_y_speed_step[3] = int(self.ids.input_y_speed_step3.text)
        conf_z_speed_step[3] = int(self.ids.input_z_speed_step3.text)
        conf_bed_pos_step[3] = 1 if self.ids.bt_bed_pos3.text == "UP" else 0

        conf_x_speed_step[4] = int(self.ids.input_x_speed_step4.text)
        conf_y_speed_step[4] = int(self.ids.input_y_speed_step4.text)
        conf_z_speed_step[4] = int(self.ids.input_z_speed_step4.text)
        conf_bed_pos_step[4] = 1 if self.ids.bt_bed_pos4.text == "UP" else 0

        conf_x_speed_step[5] = int(self.ids.input_x_speed_step5.text)
        conf_y_speed_step[5] = int(self.ids.input_y_speed_step5.text)
        conf_z_speed_step[5] = int(self.ids.input_z_speed_step5.text)
        conf_bed_pos_step[5] = 1 if self.ids.bt_bed_pos5.text == "UP" else 0

        conf_x_speed_step[6] = int(self.ids.input_x_speed_step6.text)
        conf_y_speed_step[6] = int(self.ids.input_y_speed_step6.text)
        conf_z_speed_step[6] = int(self.ids.input_z_speed_step6.text)
        conf_bed_pos_step[6] = 1 if self.ids.bt_bed_pos6.text == "UP" else 0

        conf_x_speed_step[7] = int(self.ids.input_x_speed_step7.text)
        conf_y_speed_step[7] = int(self.ids.input_y_speed_step7.text)
        conf_z_speed_step[7] = int(self.ids.input_z_speed_step7.text)
        conf_bed_pos_step[7] = 1 if self.ids.bt_bed_pos7.text == "UP" else 0

        conf_x_speed_step[8] = int(self.ids.input_x_speed_step8.text)
        conf_y_speed_step[8] = int(self.ids.input_y_speed_step8.text)
        conf_z_speed_step[8] = int(self.ids.input_z_speed_step8.text)
        conf_bed_pos_step[8] = 1 if self.ids.bt_bed_pos8.text == "UP" else 0

        conf_x_speed_step[9] = int(self.ids.input_x_speed_step9.text)
        conf_y_speed_step[9] = int(self.ids.input_y_speed_step9.text)
        conf_z_speed_step[9] = int(self.ids.input_z_speed_step9.text)
        conf_bed_pos_step[9] = 1 if self.ids.bt_bed_pos9.text == "UP" else 0

    def choice_speed(self, movement, number):
        global flag_conn_stat
        global conf_x_speed_step, conf_y_speed_step, conf_z_speed_step
        global data_base_config

        if(movement=="x"):
            for i in range(0,10):
                if(number==i):
                    if conf_x_speed_step[i] <= 5:
                        conf_x_speed_step[i] += 1

        if(movement=="y"):
            for i in range(0,10):
                if(number==i):
                    if conf_y_speed_step[i] <= 5:
                        conf_y_speed_step[i] += 1

        if(movement=="z"):
            for i in range(0,10):
                if(number==i):
                    if conf_z_speed_step[i] <= 5:
                        conf_z_speed_step[i] += 1

        conf_x_speed_step[conf_x_speed_step > 5] = 1
        conf_y_speed_step[conf_y_speed_step > 5] = 1
        conf_z_speed_step[conf_z_speed_step > 5] = 1

        self.update_text_config()

    def choice_bed(self, number):
        global flag_conn_stat
        global conf_bed_pos_step
        global data_base_config 
        for i in range (0,10):       
            if(number==i):
                if conf_bed_pos_step[i] == 1:
                    conf_bed_pos_step[i] = 0
                else:
                    conf_bed_pos_step[i] = 1

    def update_graph(self, elev=45, azim=60, roll=0):
        global val_pipe_length
        global val_pipe_diameter
        global val_pipe_thickness

        global val_x_step
        global val_y_step
        global val_z_step

        global data_base_process
        global view_camera

        view_camera = elev, azim, roll
        try:
            self.update()
            
            for i in range(0,9):
                data_base_process[0,i] = val_x_step[i]
                data_base_process[1,i] = val_y_step[i]
                data_base_process[2,i] = val_z_step[i]

            val_x_step = data_base_process[0,:]
            val_y_step = data_base_process[1,:] 
            val_z_step = data_base_process[2,:] 

            self.ids.pipe_yed_illustration.clear_widgets()

            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.fig.set_facecolor("#eeeeee")
            # self.fig.tight_layout()

            offset_length = val_x_step
            y_angle = val_y_step / 180 * np.pi
            z_angle = val_z_step / 180 * np.pi
            pipe_radius = val_pipe_diameter / 2

            Uo = np.linspace(0, 2 * np.pi, 30)
            Yo = np.linspace(0, 0, 5)
            Uo, Yo = np.meshgrid(Uo, Yo)
            Xo = pipe_radius * np.cos(Uo) - val_machine_die_radius
            Zo = pipe_radius * np.sin(Uo)
            
            X0, Y0, Z0 = self.simulate(Xo, Yo, Zo, offset_length[0], y_angle[0], z_angle[0])
            X1, Y1, Z1 = self.simulate(X0, Y0, Z0, offset_length[1], y_angle[1], z_angle[1])
            X2, Y2, Z2 = self.simulate(X1, Y1, Z1, offset_length[2], y_angle[2], z_angle[2])
            X3, Y3, Z3 = self.simulate(X2, Y2, Z2, offset_length[3], y_angle[3], z_angle[3])
            X4, Y4, Z4 = self.simulate(X3, Y3, Z3, offset_length[4], y_angle[4], z_angle[4])
            X5, Y5, Z5 = self.simulate(X4, Y4, Z4, offset_length[5], y_angle[5], z_angle[5])
            X6, Y6, Z6 = self.simulate(X5, Y5, Z5, offset_length[6], y_angle[6], z_angle[6])
            X7, Y7, Z7 = self.simulate(X6, Y6, Z6, offset_length[7], y_angle[7], z_angle[7])
            X8, Y8, Z8 = self.simulate(X7, Y7, Z7, offset_length[8], y_angle[8], z_angle[8])
            X9, Y9, Z9 = self.simulate(X8, Y8, Z8, offset_length[9], y_angle[9], z_angle[9])

            self.ax.plot_surface(X9, Y9, Z9, color='gray', alpha=1)
            self.ax.set_aspect('equal')
            self.ax.view_init(elev=view_camera[0], azim=view_camera[1], roll=view_camera[2])
            self.ids.pipe_yed_illustration.add_widget(FigureCanvasKivyAgg(self.fig))   
        except:
            toast("error update pipe ying process illustration")

    def simulate(self, prev_X, prev_Y, prev_Z, offset_length, y_angle, z_angle):
        global val_pipe_length
        global val_pipe_diameter
        global val_pipe_thickness

        global val_x_step
        global val_y_step
        global val_z_step
        
        global data_base_process
    
        pipe_radius = val_pipe_diameter / 2
        # step 1 : create straight pipe
        # straight pipe
        Ua = np.linspace(0, 2 * np.pi, 30)
        Ya = np.linspace(offset_length, 0, 5)
        Ua, Ya = np.meshgrid(Ua, Ya)
        Xa = pipe_radius * np.cos(Ua) - val_machine_die_radius
        Za = pipe_radius * np.sin(Ua)
        # combine become one object with previous mesh
        Xa = np.append(prev_X, Xa, axis=0)
        Ya = np.append(prev_Y + offset_length, Ya, axis=0)
        Za = np.append(prev_Z, Za, axis=0)

        # step 2 : create yed pipe
        # theta: poloidal angle; phi: toroidal angle 
        theta = np.linspace(0, 2 * np.pi, 30) 
        phi   = np.linspace(0, y_angle, 30) 
        theta, phi = np.meshgrid(theta, phi) 
        # torus parametrization 
        Xb = (val_machine_die_radius + pipe_radius * np.cos(theta)) * -np.cos(phi)
        Yb = (val_machine_die_radius + pipe_radius * np.cos(theta)) * -np.sin(phi)
        Zb = pipe_radius * np.sin(theta) 

        # step 3 : combine become one object
        Xc = np.append(Xa, Xb, axis=0)
        Yc = np.append(Ya, Yb, axis=0)
        Zc = np.append(Za, Zb, axis=0)

        # step 4 : rotate  object at Z axis (C axis)
        Xd = np.cos(y_angle) * Xc + np.sin(y_angle) * Yc
        Yd = -np.sin(y_angle) * Xc + np.cos(y_angle) * Yc
        Zd = Zc

        # step 5 : translate to origin, rotate  object at Y axis (B axis), translate back to previous position
        # translate
        Xe = Xd + val_machine_die_radius
        Ze = Zd
        # rotate
        Xf = np.cos(z_angle) * Xe + -np.sin(z_angle) * Ze
        Zf = np.sin(z_angle) * Xe + np.cos(z_angle) * Ze
        # translate back
        Xf = Xf - val_machine_die_radius
        Yf = Yd

        return Xf, Yf, Zf

    def reset(self):
        global val_pipe_length
        global val_pipe_diameter
        global val_pipe_thickness

        global val_x_step
        global val_y_step
        global val_z_step
        
        global data_base_process

        val_x_step = np.zeros(10)
        val_y_step = np.zeros(10)
        val_z_step = np.zeros(10)

        data_base_process = np.zeros([3, 10])

        self.ids.input_x_step0.text = str(val_x_step[0])
        self.ids.input_y_step0.text = str(val_y_step[0])
        self.ids.input_z_step0.text = str(val_z_step[0])

        self.ids.input_x_step1.text = str(val_x_step[1])
        self.ids.input_y_step1.text = str(val_y_step[1])
        self.ids.input_z_step1.text = str(val_z_step[1])

        self.ids.input_x_step2.text = str(val_x_step[2])
        self.ids.input_y_step2.text = str(val_y_step[2])
        self.ids.input_z_step2.text = str(val_z_step[2])

        self.ids.input_x_step3.text = str(val_x_step[3])
        self.ids.input_y_step3.text = str(val_y_step[3])
        self.ids.input_z_step3.text = str(val_z_step[3])

        self.ids.input_x_step4.text = str(val_x_step[4])
        self.ids.input_y_step4.text = str(val_y_step[4])
        self.ids.input_z_step4.text = str(val_z_step[4])

        self.ids.input_x_step5.text = str(val_x_step[5])
        self.ids.input_y_step5.text = str(val_y_step[5])
        self.ids.input_z_step5.text = str(val_z_step[5])

        self.ids.input_x_step6.text = str(val_x_step[6])
        self.ids.input_y_step6.text = str(val_y_step[6])
        self.ids.input_z_step6.text = str(val_z_step[6])

        self.ids.input_x_step7.text = str(val_x_step[7])
        self.ids.input_y_step7.text = str(val_y_step[7])
        self.ids.input_z_step7.text = str(val_z_step[7])

        self.ids.input_x_step8.text = str(val_x_step[8])
        self.ids.input_y_step8.text = str(val_y_step[8])
        self.ids.input_z_step8.text = str(val_z_step[8])

        self.ids.input_x_step9.text = str(val_x_step[9])
        self.ids.input_y_step9.text = str(val_y_step[9])
        self.ids.input_z_step9.text = str(val_z_step[9]) 

        self.update_graph()

    def save(self):
        try:
            name_file = "\data\\" + self.ids.input_file_name.text + ".gcode"
            name_file_now = datetime.now().strftime("\data\%d_%m_%Y_%H_%M_%S.gcode")
            cwd = os.getcwd()
            if self.ids.input_file_name.text == "":
                disk = cwd + name_file_now
            else:
                disk = cwd + name_file
            print(disk)
            data_base_save = np.vstack((data_base_process, data_base_config))
            print(data_base_save)
            with open(disk,"wb") as f:
                np.savetxt(f, data_base_save.T, fmt="%.3f",delimiter="\t",header="x [mm] \t y [mm] \t Plane [mm] \t x Speed \t y Speed \t Plane Speed \t Bed Pos")
            print("sucessfully save data")
            toast("sucessfully save data")
        except Exception as e:
            print(e)
            print("error saving data")
            toast("error saving data")
                
    def screen_main_menu(self):
        self.screen_manager.current = 'screen_main_menu'

    def screen_pipe_setting(self):
        self.screen_manager.current = 'screen_pipe_setting'

    def screen_machine_setting(self):
        self.screen_manager.current = 'screen_machine_setting'

    def screen_advanced_setting(self):
        self.screen_manager.current = 'screen_advanced_setting'

    def screen_operate_auto(self):
        self.screen_manager.current = 'screen_operate_auto'

    def screen_compile(self):
        self.screen_manager.current = 'screen_compile'

    def exec_shutdown(self):
        os.system("shutdown /s /t 1") #for windows os
        toast("shutting down system")
        # os.system("shutdown -h 1")

class RootScreen(ScreenManager):
    pass

class PipeyingCNCApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self):
        self.theme_cls.colors = colors
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.accent_palette = "Gray"
        self.icon = 'asset/logo.png'
        Window.fullscreen = 'auto'
        Window.borderless = False
        # Window.size = 900, 1440
        # Window.size = 450, 720
        # Window.allow_screensaver = True

        Builder.load_file('main.kv')
        return RootScreen()

if __name__ == '__main__':
    PipeyingCNCApp().run()