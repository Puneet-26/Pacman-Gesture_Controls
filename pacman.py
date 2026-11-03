"""
pacman.py
Main application entry point for Pac-Man with real-time hand gesture controls.
Supports both Keyboard Mode and Gesture Mode (OpenCV + MediaPipe webcam tracking).
"""

import os
import sys
import time
from typing import Optional

# Center Pygame display window on desktop
os.environ['SDL_VIDEO_CENTERED'] = '1'

import pygame
from model.board_definition import BoardDefinition
from model.level_config import LevelConfig
from settings import *
from levels.level_content_initializer import LevelContentInitializer
from model.direction import Direction
from gesture_controller import GestureController, GestureType


class Game:
    """Coordinates Pac-Man game mechanics, display rendering, and multi-mode inputs."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Pac-Man (Gesture & Keyboard Controls)")
        self.screen = pygame.display.set_mode(RESOLUTION)
        self.timer = pygame.time.Clock()
        self.game_engine = self.init()
        self.game_start_sfx = pygame.mixer.Sound('media/game_start.wav')

        # Control Modes: "KEYBOARD" or "GESTURE"
        self.control_mode = "KEYBOARD"
        self.gesture_controller: Optional[GestureController] = None
        self.showing_mode_select = True
        self.selected_menu_index = 0

        # Notification & HUD state
        self.status_message: Optional[str] = None
        self.status_message_expiry: float = 0.0
        ARCADE_FONT = 'consolas,couriernew,arial'
        self.ui_font = pygame.font.SysFont(ARCADE_FONT, 16)
        self.title_font = pygame.font.SysFont(ARCADE_FONT, 38, bold=True)
        self.menu_font = pygame.font.SysFont(ARCADE_FONT, 21, bold=True)
        self.sub_font = pygame.font.SysFont(ARCADE_FONT, 15)

    def init(self):
        """Initializes a fresh level, maze board, and game engine."""
        board = BOARD.copy()
        board_definition = BoardDefinition(board)
        level_1 = LevelConfig(wall_color='blue', gate_color='white',
                              board_definition=board_definition, power_up_limit=POWER_UP_LIMIT)
        level_init = LevelContentInitializer(level_1, self.screen)
        return level_init.init_game_engine()

    def set_notification(self, message: str, duration: float = 3.0):
        """Displays temporary notification message on game screen."""
        self.status_message = message
        self.status_message_expiry = time.time() + duration

    def set_control_mode(self, mode: str) -> bool:
        """
        Safely switches active control mode.
        Starts or releases webcam resources as needed.
        """
        if mode == "GESTURE":
            if self.gesture_controller and self.gesture_controller.is_active():
                self.control_mode = "GESTURE"
                self.set_notification("GESTURE MODE ACTIVE [Webcam Tracking]")
                return True

            print("[Pac-Man] Starting Gesture Controller...")
            self.gesture_controller = GestureController(show_preview=True)
            if self.gesture_controller.start():
                self.control_mode = "GESTURE"
                self.set_notification("GESTURE MODE ACTIVE [Webcam Ready]")
                return True
            else:
                err = self.gesture_controller.error_message or "Could not initialize webcam."
                self.set_notification(f"Camera Error: {err} -> Fallback to Keyboard", duration=4.0)
                print(f"[Pac-Man] {err}")
                self.control_mode = "KEYBOARD"
                self.gesture_controller = None
                return False
        else:
            # Switching to KEYBOARD mode
            if self.gesture_controller:
                self.gesture_controller.stop()
                self.gesture_controller = None
            self.control_mode = "KEYBOARD"
            self.set_notification("KEYBOARD MODE ACTIVE [Arrow Keys / WASD]")
            return True

    def toggle_control_mode(self):
        """Toggles between Keyboard Mode and Gesture Mode."""
        new_mode = "GESTURE" if self.control_mode == "KEYBOARD" else "KEYBOARD"
        self.set_control_mode(new_mode)

    def process_gesture_input(self):
        """Polls latest non-blocking commands from the gesture controller."""
        if self.control_mode != "GESTURE" or not self.gesture_controller:
            return

        # Check for camera errors during gameplay
        if self.gesture_controller.error_message:
            self.set_notification(f"[!] {self.gesture_controller.error_message} -> Reverted to Keyboard", duration=4.0)
            self.set_control_mode("KEYBOARD")
            return

        cmd = self.gesture_controller.get_command()
        if cmd is None:
            return

        if cmd == GestureType.LEFT:
            self.game_engine.set_direction(Direction.LEFT)
        elif cmd == GestureType.RIGHT:
            self.game_engine.set_direction(Direction.RIGHT)
        elif cmd == GestureType.UP:
            self.game_engine.set_direction(Direction.UP)
        elif cmd == GestureType.DOWN:
            self.game_engine.set_direction(Direction.DOWN)
        elif cmd == GestureType.PAUSE:
            self.game_engine.pause = True
        elif cmd == GestureType.RESUME:
            self.game_engine.pause = False

    def check_events(self):
        """Handles Pygame input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.cleanup()
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                # Mode selection screen inputs
                if self.showing_mode_select:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        self.selected_menu_index = (self.selected_menu_index - 1) % 2
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        self.selected_menu_index = (self.selected_menu_index + 1) % 2
                    elif event.key in (pygame.K_1, pygame.K_k):
                        self.selected_menu_index = 0
                        self.set_control_mode("KEYBOARD")
                        self.showing_mode_select = False
                    elif event.key in (pygame.K_2, pygame.K_g):
                        self.selected_menu_index = 1
                        self.set_control_mode("GESTURE")
                        self.showing_mode_select = False
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        chosen_mode = "KEYBOARD" if self.selected_menu_index == 0 else "GESTURE"
                        self.set_control_mode(chosen_mode)
                        self.showing_mode_select = False
                    elif event.key == pygame.K_ESCAPE:
                        self.cleanup()
                        pygame.quit()
                        sys.exit()
                    continue

                # In-game mode toggle: 'M' key
                if event.key == pygame.K_m:
                    self.toggle_control_mode()
                    continue

                # In-game pause toggle: 'P' or 'ESCAPE'
                if event.key in (pygame.K_p, pygame.K_ESCAPE):
                    self.game_engine.pause = not self.game_engine.pause
                    continue

                # Keyboard Directional Controls (Arrow keys & WASD)
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    self.game_engine.set_direction(Direction.LEFT)
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.game_engine.set_direction(Direction.RIGHT)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.game_engine.set_direction(Direction.DOWN)
                elif event.key in (pygame.K_UP, pygame.K_w):
                    self.game_engine.set_direction(Direction.UP)

                # Restart game when over
                if event.key == pygame.K_SPACE and self.game_engine.game_over:
                    self.game_start_sfx.play()
                    self.game_engine = self.init()

            if event.type == GHOST_EATEN_EVENT:
                self.game_engine.play_ghost_runsaway_sound()
            if event.type == PLAYER_EATEN_EVENT:
                self.game_engine.play_player_eaten_sound()

    def update(self):
        """Advances game simulation frame."""
        self.timer.tick(FPS)

        if not self.showing_mode_select:
            # Poll gesture inputs
            self.process_gesture_input()
            # Advance game physics and ghost AI
            self.game_engine.tick()

        # Render HUD overlays
        self.render_hud()
        pygame.display.flip()

    def draw(self):
        """Clears screen canvas."""
        self.screen.fill([12, 2, 25])

    def render_hud(self):
        """Renders mode badges, notification toasts, and mode selection menu."""
        now = time.time()

        if self.showing_mode_select:
            self.render_mode_select_screen()
            return

        # 1. Bottom status bar
        bar_y = self.screen.get_height() - 24
        mode_color = (0, 255, 200) if self.control_mode == "GESTURE" else (255, 220, 0)
        mode_text = self.ui_font.render(f"[{self.control_mode} MODE]", True, mode_color)
        self.screen.blit(mode_text, (15, bar_y))

        hint_text = self.ui_font.render("Press 'M': Toggle Mode | 'P': Pause", True, (180, 180, 180))
        self.screen.blit(hint_text, (self.screen.get_width() - hint_text.get_width() - 15, bar_y))

        # 2. Notification Toast
        if self.status_message and now < self.status_message_expiry:
            toast_surf = self.ui_font.render(self.status_message, True, (255, 255, 255))
            toast_w = toast_surf.get_width() + 30
            toast_h = 36
            toast_x = (self.screen.get_width() - toast_w) // 2
            toast_y = 12

            bg_surf = pygame.Surface((toast_w, toast_h), pygame.SRCALPHA)
            bg_surf.fill((10, 10, 20, 220))
            pygame.draw.rect(bg_surf, mode_color, (0, 0, toast_w, toast_h), 2, border_radius=6)
            self.screen.blit(bg_surf, (toast_x, toast_y))
            self.screen.blit(toast_surf, (toast_x + 15, toast_y + 8))

    def render_mode_select_screen(self):
        """Renders minimal, classic arcade-style title screen."""
        # Pure black background
        self.screen.fill((0, 0, 0))

        cx = self.screen.get_width() // 2
        COLOR_YELLOW = (255, 222, 0)
        COLOR_WHITE = (255, 255, 255)
        COLOR_GREY = (140, 140, 140)
        COLOR_MUTED = (110, 110, 110)

        # 1. Title: PAC-MAN VISION in bold yellow
        title_surf = self.title_font.render("PAC-MAN VISION", True, COLOR_YELLOW)
        self.screen.blit(title_surf, (cx - title_surf.get_width() // 2, 170))

        # 2. Vertically stacked menu options
        options = [
            "Keyboard Mode (Arrows / WASD)",
            "Gesture Mode (Webcam Hand Tracking)"
        ]

        start_y = 300
        line_spacing = 45

        for idx, opt_text in enumerate(options):
            is_selected = (idx == self.selected_menu_index)
            text_color = COLOR_YELLOW if is_selected else COLOR_WHITE
            opt_surf = self.menu_font.render(opt_text, True, text_color)
            opt_x = cx - opt_surf.get_width() // 2
            opt_y = start_y + idx * line_spacing

            self.screen.blit(opt_surf, (opt_x, opt_y))

            # Small yellow circular indicator beside selected option
            if is_selected:
                circle_x = opt_x - 20
                circle_y = opt_y + opt_surf.get_height() // 2
                pygame.draw.circle(self.screen, COLOR_YELLOW, (circle_x, circle_y), 5)

        # 3. Navigation Instructions
        nav_text_1 = self.sub_font.render("UP / DOWN: Navigate", True, COLOR_GREY)
        nav_text_2 = self.sub_font.render("ENTER / SPACE: Select", True, COLOR_GREY)
        self.screen.blit(nav_text_1, (cx - nav_text_1.get_width() // 2, 435))
        self.screen.blit(nav_text_2, (cx - nav_text_2.get_width() // 2, 465))

        # 4. Gesture Information
        g_title = self.sub_font.render("Point Index: UP / DOWN / LEFT / RIGHT", True, COLOR_MUTED)
        g_actions = self.sub_font.render("Fist: Pause    Open Palm: Resume", True, COLOR_MUTED)
        g_switch = self.sub_font.render("Press 'M' to switch modes anytime during gameplay", True, COLOR_MUTED)

        self.screen.blit(g_title, (cx - g_title.get_width() // 2, 570))
        self.screen.blit(g_actions, (cx - g_actions.get_width() // 2, 598))
        self.screen.blit(g_switch, (cx - g_switch.get_width() // 2, 626))

    def cleanup(self):
        """Safely terminates controllers and camera."""
        if self.gesture_controller:
            self.gesture_controller.stop()
            self.gesture_controller = None

    def run(self):
        """Main game loop."""
        try:
            self.game_start_sfx.play()
            while True:
                self.check_events()
                self.draw()
                self.update()
        except KeyboardInterrupt:
            print("\n[Pac-Man] Exiting...")
        finally:
            self.cleanup()


if __name__ == '__main__':
    game = Game()
    game.run()
