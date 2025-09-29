from __future__ import annotations
import os
import collections
from typing import List, Tuple, Optional, Dict
import math
import pygame
import torch
import torch.nn as nn
import numpy as np

import config
from .camera import Camera
from engine.agent_registry import COL_ALIVE, COL_TEAM, COL_HP, COL_X, COL_Y, COL_UNIT, COL_HP_MAX, COL_VISION, COL_ATK
from engine.ray_engine.raycast_64 import raycast64_firsthit, DIRS64
from engine.ray_engine.raycast_firsthit import build_unit_map

# --- Constants & Configuration ---
FONT_NAME = "consolas"

# New, more vibrant and distinct color palette
COLORS = {
    "bg": (20, 22, 28), "hud_bg": (12, 14, 18), "side_bg": (18, 20, 26),
    "grid": (40, 42, 48), "border": (70, 74, 82), "wall": (90, 94, 102),
    "empty": (24, 26, 32),
    
    # Red Team (Crimson/Fire)
    "red_soldier":  (231, 76, 60),   # Alizarin Crimson
    "red_archer":   (211, 84, 0),    # Pumpkin Orange
    "red":          (231, 76, 60),

    # Blue Team (Sky/Ocean)
    "blue_soldier": (52, 152, 219),  # Peter River Blue
    "blue_archer":  (22, 160, 133),   # Green Sea
    "blue":         (52, 152, 219),
    
    "archer_glyph": (245, 230, 90), "marker": (242, 228, 92),
    "text": (230, 230, 230), "text_dim": (180, 186, 194),
    "green": (46, 204, 113), "warn": (243, 156, 18),
    "bar_bg": (38, 42, 48), "bar_fg": (46, 204, 113),
    "graph_red": (231, 76, 60, 150), "graph_blue": (52, 152, 219, 150),
    "graph_grid": (60, 60, 70), "pause_text": (241, 196, 15)
}
OVERLAYS = {
    "heal": (46, 204, 113, 60), "cp": (210, 210, 230, 48),
    "outline_red": (231, 76, 60, 160), "outline_blue": (52, 152, 219, 160),
    "outline_neutral": (160, 160, 170, 120),
    "threat_enemy": (231, 76, 60), "threat_ally": (52, 152, 219),
    "vision_range": (180, 180, 180, 40)
}
RAY_COLORS = {
    "ally": (52, 152, 219), "enemy": (231, 76, 60),
    "wall": (180, 180, 180), "empty": (100, 100, 110)
}

# --- Utility Functions ---
def _center_window():
    os.environ.setdefault("SDL_VIDEO_CENTERED", "1")

def _get_model_summary(model: nn.Module) -> str:
    name = model.__class__.__name__.lower()
    if "transformer" in name:
        try:
            # Attempt to get transformer-specific details, fallback gracefully
            d = model.embed_dim
            num_cross = 1 if hasattr(model, 'cross_attention') else 0
            num_self = 1 if hasattr(model, 'self_attention') else 0
            return f"Transformer(d={d}, Cross={num_cross}, Self={num_self})"
        except Exception: return "TransformerBrain"
    try:
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        dims = [linears[0].in_features] + [m.out_features for m in linears]
        return "→".join(map(str, dims))
    except Exception: return "Unknown"

def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TextCache:
    def __init__(self):
        self.fonts = {
            13: self._mk_font(13), 16: self._mk_font(16), 18: self._mk_font(18)
        }
        self.cache = {}
    def _mk_font(self, sz):
        try: return pygame.font.SysFont(FONT_NAME, sz)
        except: return pygame.font.Font(None, sz)
    def render(self, text, size, color, aa=True):
        key = (text, size, color, aa)
        if key not in self.cache:
            self.cache[key] = self.fonts[size].render(text, aa, color)
        return self.cache[key]

# --- UI Components ---
class LayoutManager:
    def __init__(self, viewer): self.viewer = viewer
    def side_width(self): return max(320, min(420, int(self.viewer.Wpix * 0.27)))
    def world_rect(self):
        m = self.viewer.margin
        return pygame.Rect(m, m, max(64, self.viewer.Wpix - self.side_width() - 3*m), max(64, self.viewer.Hpix - 126 - 2*m))
    def side_rect(self):
        m, side_w = self.viewer.margin, self.side_width()
        return pygame.Rect(self.viewer.Wpix - side_w - m, m, side_w, max(64, self.viewer.Hpix - 126 - 2*m))
    def hud_rect(self): return pygame.Rect(0, self.viewer.Hpix - 126, self.viewer.Wpix, 126)

class WorldRenderer:
    def __init__(self, viewer, grid, registry):
        self.viewer, self.grid, self.registry = viewer, grid, registry
        self.cam = viewer.cam
        self.static_surf = None
        self._zone_cache = {"heal_tiles": [], "cp_rects": []}

    def build_static_cache(self, engine):
        self.static_surf = None
        self._zone_cache = {"heal_tiles": [], "cp_rects": []}
        zones = getattr(engine, "zones", None)
        if zones:
            if getattr(zones, "heal_mask", None) is not None:
                ys, xs = torch.nonzero(zones.heal_mask, as_tuple=True)
                self._zone_cache["heal_tiles"] = list(zip(xs.cpu().tolist(), ys.cpu().tolist()))
            for m in getattr(zones, "cp_masks", []):
                ys, xs = torch.nonzero(m, as_tuple=True)
                if xs.numel() > 0:
                    self._zone_cache["cp_rects"].append((xs.min().item(), ys.min().item(), xs.max().item()+1, ys.max().item()+1))

    def _draw_static_background(self):
        wrect = self.viewer.layout.world_rect()
        self.static_surf = pygame.Surface(wrect.size)
        self.static_surf.fill(COLORS["bg"])
        H, W = self.grid.shape[1], self.grid.shape[2]
        occ_np = self.grid[0].cpu().numpy()
        for y in range(H):
            for x in range(W):
                occ = occ_np[y,x]
                if occ in {0, 1}:
                    color = COLORS["empty"] if occ == 0 else COLORS["wall"]
                    cx, cy = self.cam.world_to_screen(x, y)
                    pygame.draw.rect(self.static_surf, color, (cx, cy, self.cam.cell_px, self.cam.cell_px))
        overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)
        for x, y in self._zone_cache["heal_tiles"]:
            cx, cy = self.cam.world_to_screen(x,y)
            pygame.draw.rect(overlay, OVERLAYS["heal"], (cx, cy, self.cam.cell_px, self.cam.cell_px))
        self.static_surf.blit(overlay, (0,0))

    def draw(self, surf, state_data):
        wrect = self.viewer.layout.world_rect()
        if self.static_surf is None or self.static_surf.get_size() != wrect.size: self._draw_static_background()
        surf.blit(self.static_surf, wrect.topleft)
        
        c = self.cam.cell_px
        cp_overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)
        for x0, y0, x1, y1 in self._zone_cache["cp_rects"]:
            patch = state_data["occ_np"][y0:y1, x0:x1]
            red_cnt, blue_cnt = (patch==2).sum(), (patch==3).sum()
            if red_cnt > blue_cnt: b_col, label = OVERLAYS["outline_red"], ("R", COLORS["red"])
            elif blue_cnt > red_cnt: b_col, label = OVERLAYS["outline_blue"], ("B", COLORS["blue"])
            else: b_col, label = OVERLAYS["outline_neutral"], ("–", COLORS["text_dim"])
            cx0, cy0 = self.cam.world_to_screen(x0,y0); cx1, cy1 = self.cam.world_to_screen(x1,y1)
            rect = pygame.Rect(cx0, cy0, cx1-cx0, cy1-cy0)
            pygame.draw.rect(cp_overlay, b_col, rect, max(1, c//2))
            lab_surf = self.viewer.text_cache.render(label[0], 13, label[1])
            cp_overlay.blit(lab_surf, lab_surf.get_rect(center=rect.center))
        surf.blit(cp_overlay, wrect.topleft)

        for i in state_data["alive_indices"]:
            if i not in state_data["agent_map"]: continue
            x, y, unit, team = state_data["agent_map"][i]
            color_key = f"{'red' if team == 2.0 else 'blue'}_{'archer' if unit == 2.0 else 'soldier'}"
            color = COLORS[color_key]
            cx, cy = self.cam.world_to_screen(x,y)
            pygame.draw.rect(surf, color, (wrect.x+cx, wrect.y+cy, c,c))
            if unit == 2.0 and c > 4:
                pygame.draw.circle(surf, COLORS["archer_glyph"], (wrect.x+cx+c//2, wrect.y+cy+c//2), max(2, c//2-1), max(1, c//6))
        
        if self.viewer.battle_view_enabled: self._draw_hp_bars(surf, wrect, c, state_data)
        if self.viewer.threat_vision_mode: self._draw_threat_vision(surf, wrect, c, state_data)
        if self.viewer.show_grid and c >= 6: self._draw_grid_lines(surf, wrect, c)
        self._draw_markers(surf, wrect, c, state_data["id_np"])
        if self.viewer.show_rays: self._draw_rays(surf, wrect, c, state_data)

    def _draw_hp_bars(self, surf, wrect, c, state_data):
        if c < 8: return
        hp_bar_surf = pygame.Surface(wrect.size, pygame.SRCALPHA)
        for aid in state_data["alive_indices"]:
            x, y, _, _ = state_data["agent_map"][aid]
            hp_max = self.registry.agent_data[aid, COL_HP_MAX].item()
            if hp_max > 0:
                hp_ratio = self.registry.agent_data[aid, COL_HP].item() / hp_max
            else:
                hp_ratio = 0.0
            cx, cy = self.cam.world_to_screen(x,y)
            
            bar_w, bar_h = c, max(1, c // 8)
            bar_y = cy - bar_h - 2
            
            if 0 <= bar_y < wrect.height and 0 <= cx < wrect.width:
                pygame.draw.rect(hp_bar_surf, (50,50,50), (cx, bar_y, bar_w, bar_h))
                pygame.draw.rect(hp_bar_surf, COLORS["green"], (cx, bar_y, bar_w * hp_ratio, bar_h))
        surf.blit(hp_bar_surf, wrect.topleft)
        
    def _draw_threat_vision(self, surf, wrect, c, state_data):
        aid = self.viewer.selected_id
        if aid is None or aid not in state_data["agent_map"]: return
        
        my_x, my_y, _, my_team = state_data["agent_map"][aid]
        my_cx, my_cy = self.cam.world_to_screen(my_x, my_y)

        vision_range = self.registry.agent_data[aid, COL_VISION].item()
        vision_px = vision_range * c
        
        overlay = pygame.Surface(wrect.size, pygame.SRCALPHA)
        center_px = (my_cx + c//2, my_cy + c//2)
        pygame.draw.circle(overlay, OVERLAYS["vision_range"], center_px, vision_px)

        for other_aid, (ox, oy, _, o_team) in state_data["agent_map"].items():
            if aid == other_aid: continue
            dist = np.hypot(ox - my_x, oy - my_y)
            if dist <= vision_range:
                o_cx, o_cy = self.cam.world_to_screen(ox, oy)
                hp_max = self.registry.agent_data[other_aid, COL_HP_MAX].item()
                hp_ratio = (self.registry.agent_data[other_aid, COL_HP].item() / hp_max) if hp_max > 0 else 0
                if o_team != my_team:
                    radius = int(c * 0.7 * (0.5 + hp_ratio * 0.5))
                    alpha = int(50 + 150 * hp_ratio)
                    pygame.draw.circle(overlay, (*OVERLAYS["threat_enemy"], alpha), (o_cx + c//2, o_cy + c//2), radius)
                else:
                    pygame.draw.circle(overlay, (*OVERLAYS["threat_ally"], 100), (o_cx + c//2, o_cy + c//2), int(c*0.4), 1)
        surf.blit(overlay, wrect.topleft)

    def _draw_grid_lines(self, surf, wrect, c):
        ax, ay = self.cam.world_to_screen(0,0)
        off_x, off_y = (c - (ax % c)) % c, (c - (ay % c)) % c
        x, y = wrect.x + off_x, wrect.y + off_y
        while x < wrect.right: pygame.draw.line(surf, COLORS["grid"], (x, wrect.y), (x, wrect.bottom)); x += c
        while y < wrect.bottom: pygame.draw.line(surf, COLORS["grid"], (wrect.x, y), (wrect.right, y)); y += c
        pygame.draw.rect(surf, COLORS["border"], wrect, 2)
    
    def _draw_markers(self, surf, wrect, c, id_np):
        for aid in self.viewer.marked:
            pos = np.argwhere(id_np == aid)
            if pos.size > 0:
                y, x = pos[0]
                cx, cy = self.cam.world_to_screen(x,y)
                pygame.draw.rect(surf, COLORS["marker"], (wrect.x+cx, wrect.y+cy, c,c), max(1, c//8))
    
    def _draw_rays(self, surf, wrect, c, state_data):
        aid = self.viewer.selected_id
        if aid is None or aid not in state_data["agent_map"]: return

        agent_x, agent_y, _, my_team = state_data["agent_map"][aid]
        start_pos_screen = (wrect.x + self.cam.world_to_screen(agent_x, agent_y)[0] + c // 2,
                            wrect.y + self.cam.world_to_screen(agent_x, agent_y)[1] + c // 2)

        vision_range = int(self.registry.agent_data[aid, COL_VISION].item())
        occ_grid = state_data["occ_np"]
        H, W = occ_grid.shape

        num_rays_to_draw = 32
        for i in range(num_rays_to_draw):
            angle = i * (2 * math.pi / num_rays_to_draw)
            dx, dy = math.cos(angle), math.sin(angle)

            end_x, end_y = agent_x, agent_y
            color = RAY_COLORS["empty"]

            # --- Manual Ray Step ---
            for step in range(1, vision_range + 1):
                test_x = int(round(agent_x + dx * step))
                test_y = int(round(agent_y + dy * step))

                # Check bounds
                if not (0 <= test_x < W and 0 <= test_y < H):
                    end_x, end_y = test_x, test_y
                    break

                occupant = occ_grid[test_y, test_x]
                if occupant != 0: # Hit something
                    end_x, end_y = test_x, test_y
                    if occupant == 1: # Wall
                        color = RAY_COLORS["wall"]
                    else: # Agent
                        hit_team = occupant
                        color = RAY_COLORS["enemy"] if my_team != hit_team else RAY_COLORS["ally"]
                    break
            else: # No hit within range
                end_x = agent_x + dx * vision_range
                end_y = agent_y + dy * vision_range

            # Draw the final line
            end_pos_world = self.cam.world_to_screen(end_x, end_y)
            end_pos_screen = (wrect.x + end_pos_world[0] + c // 2,
                              wrect.y + end_pos_world[1] + c // 2)
            pygame.draw.line(surf, color, start_pos_screen, end_pos_screen, 1)

class HudPanel:
    def __init__(self, viewer, stats):
        self.viewer, self.stats = viewer, stats
        self.score_history = collections.deque(maxlen=1000)

    def update(self):
        self.score_history.append(self.stats.red.score - self.stats.blue.score)

    def draw(self, surf, state_data):
        hud = self.viewer.layout.hud_rect()
        surf.fill(COLORS["hud_bg"], hud)
        y, x = hud.y + 8, 12
        
        pause_str = "[ PAUSED ]" if self.viewer.paused else f"[ {self.viewer.speed_multiplier}x ]"
        surf.blit(self.viewer.text_cache.render(f"Tick {self.stats.tick} {pause_str}", 16, COLORS["pause_text"] if self.viewer.paused else COLORS["text"]), (x, y))

        self._draw_team_stats(surf, y, x, state_data)
        self._draw_score_graph(surf, hud)
        self.viewer.minimap.draw(surf, hud, state_data)

    def _draw_team_stats(self, surf, y, x, state_data):
        r_alive = sum(1 for _, _, _, team in state_data["agent_map"].values() if team == 2.0)
        b_alive = len(state_data["agent_map"]) - r_alive
        rs_alive = sum(1 for _, _, unit, team in state_data["agent_map"].values() if team == 2.0 and unit == 1.0)
        ra_alive = r_alive - rs_alive
        bs_alive = sum(1 for _, _, unit, team in state_data["agent_map"].values() if team == 3.0 and unit == 1.0)
        ba_alive = b_alive - bs_alive

        red_str = f"Red  S:{self.stats.red.score:6.1f} CP:{self.stats.red.cp_points:4.1f} K:{self.stats.red.kills:3d} D:{self.stats.red.deaths:3d} Alive:{r_alive:3d} (S:{rs_alive} A:{ra_alive})"
        blue_str = f"Blue S:{self.stats.blue.score:6.1f} CP:{self.stats.blue.cp_points:4.1f} K:{self.stats.blue.kills:3d} D:{self.stats.blue.deaths:3d} Alive:{b_alive:3d} (S:{bs_alive} A:{ba_alive})"
        surf.blit(self.viewer.text_cache.render(red_str, 16, COLORS["red"]), (x, y + 24))
        surf.blit(self.viewer.text_cache.render(blue_str, 16, COLORS["blue"]), (x, y + 48))

    def _draw_score_graph(self, surf, hud):
        graph_rect = pygame.Rect(hud.right - 540, hud.y + 10, 300, hud.height - 20)
        pygame.draw.rect(surf, COLORS["bg"], graph_rect)
        if len(self.score_history) < 2: return
        scores = np.array(self.score_history)
        max_abs_score = np.max(np.abs(scores)) or 1
        norm_scores = scores / max_abs_score
        points = [(graph_rect.x + (i/(len(scores)-1))*graph_rect.width, graph_rect.centery - s*(graph_rect.height/2.2)) for i, s in enumerate(norm_scores)]
        for i in range(1, 4): pygame.draw.line(surf, COLORS["graph_grid"], (graph_rect.x, graph_rect.y + i*graph_rect.height/4), (graph_rect.right, graph_rect.y + i*graph_rect.height/4))
        
        red_poly_points = [(p[0], p[1]) for p in points if p[1] < graph_rect.centery]
        if red_poly_points:
            red_poly = [(graph_rect.x, graph_rect.centery)] + red_poly_points + [(red_poly_points[-1][0], graph_rect.centery)]
            pygame.draw.polygon(surf, COLORS["graph_red"], red_poly)

        blue_poly_points = [(p[0], p[1]) for p in points if p[1] > graph_rect.centery]
        if blue_poly_points:
            blue_poly = [(graph_rect.x, graph_rect.centery)] + blue_poly_points + [(blue_poly_points[-1][0], graph_rect.centery)]
            pygame.draw.polygon(surf, COLORS["graph_blue"], blue_poly)
            
        pygame.draw.aalines(surf, COLORS["text"], False, points)
        pygame.draw.rect(surf, COLORS["border"], graph_rect, 1)
        surf.blit(self.viewer.text_cache.render("Score Lead", 13, COLORS["text_dim"]), (graph_rect.x + 5, graph_rect.y + 2))

class SidePanel:
    def __init__(self, viewer, registry):
        self.viewer, self.registry = viewer, registry

    def _draw_legend(self, surf, x, y):
        y += 20
        surf.blit(self.viewer.text_cache.render("Legend & Controls", 18, COLORS["text"]), (x, y)); y += 30

        # Colors & Units Legend
        legend_items = {
            "Red Soldier": COLORS["red_soldier"],
            "Red Archer": COLORS["red_archer"],
            "Blue Soldier": COLORS["blue_soldier"],
            "Blue Archer": COLORS["blue_archer"],
        }
        
        for label, color in legend_items.items():
            pygame.draw.rect(surf, color, (x, y, 15, 15))
            surf.blit(self.viewer.text_cache.render(label, 13, COLORS["text_dim"]), (x + 25, y)); y += 22

        y += 20 # Spacer
        
        # Controls Legend
        controls = [
            "WASD / Arrows: Pan Camera",
            "Mouse Wheel: Zoom",
            "Click Agent: Inspect",
            "SPACE: Pause Simulation",
            "+/-: Change Speed",
            "R: Toggle Agent Rays",
            "T: Toggle Threat Vision",
            "B: Toggle HP Bars",
            "S: Save Selected Brain",
            "F11: Toggle Fullscreen",
        ]
        
        for line in controls:
            surf.blit(self.viewer.text_cache.render(line, 13, COLORS["text_dim"]), (x, y)); y += 18
            
        return y
    
    def draw(self, surf, state_data):
        srect = self.viewer.layout.side_rect()
        surf.fill(COLORS["side_bg"], srect)
        pygame.draw.rect(surf, COLORS["border"], srect, 2)
        pad, y = 12, srect.y + 12
        x = srect.x + pad
        
        surf.blit(self.viewer.text_cache.render("Agent Inspector", 18, COLORS["text"]), (x, y)); y += 30
        aid = self.viewer.selected_id
        
        if aid is None:
            surf.blit(self.viewer.text_cache.render("Click an agent to inspect.", 13, COLORS["text_dim"]), (x, y)); y += 30
        elif aid not in state_data["agent_map"]:
            surf.blit(self.viewer.text_cache.render(f"ID: {aid} (Dead)", 13, COLORS["warn"]), (x, y)); y += 30
        else:
            surf.blit(self.viewer.text_cache.render(f"ID: {aid}", 16, COLORS["green"]), (x, y)); y += 24
            
            agent_data = self.registry.agent_data[aid]
            pos = (int(agent_data[COL_X].item()), int(agent_data[COL_Y].item()))
            hp, hp_max = agent_data[COL_HP].item(), agent_data[COL_HP_MAX].item()
            atk, vision = agent_data[COL_ATK].item(), agent_data[COL_VISION].item()
            hp_ratio = hp / hp_max if hp_max > 0 else 0
            
            surf.blit(self.viewer.text_cache.render(f"Pos: ({pos[0]}, {pos[1]})", 13, COLORS["text_dim"]), (x, y)); y += 18
            bar_w = srect.width - 2 * pad
            pygame.draw.rect(surf, COLORS["bar_bg"], (x, y, bar_w, 10))
            pygame.draw.rect(surf, COLORS["bar_fg"], (x, y, bar_w * hp_ratio, 10))
            y += 14
            surf.blit(self.viewer.text_cache.render(f"HP: {hp:.2f} / {hp_max:.2f}", 13, COLORS["text_dim"]), (x, y)); y += 20
            surf.blit(self.viewer.text_cache.render(f"Attack: {atk:.2f}", 13, COLORS["text_dim"]), (x, y)); y += 18
            surf.blit(self.viewer.text_cache.render(f"Vision: {vision} cells", 13, COLORS["text_dim"]), (x, y)); y += 22

            brain = self.registry.brains[aid]
            if brain:
                surf.blit(self.viewer.text_cache.render(f"Model: {_get_model_summary(brain)}", 13, COLORS["text_dim"]), (x, y)); y += 18
                surf.blit(self.viewer.text_cache.render(f"Params: {_param_count(brain):,}", 13, COLORS["text_dim"]), (x, y)); y += 18

        # Draw the legend regardless of selection
        pygame.draw.line(surf, COLORS["border"], (srect.x, y + 10), (srect.right, y + 10), 2)
        self._draw_legend(surf, x, y)


class Minimap:
    def __init__(self, viewer):
        self.viewer = viewer
        self.grid_w = viewer.grid.shape[2]
        self.grid_h = viewer.grid.shape[1]

    def draw(self, surf, hud_rect, state_data):
        map_w = 200
        map_h = int(map_w * (self.grid_h / self.grid_w))
        map_rect = pygame.Rect(hud_rect.right - map_w - 20, hud_rect.y + (hud_rect.height - map_h) // 2, map_w, map_h)
        
        surf.fill(COLORS["empty"], map_rect)
        
        # Draw agents
        for x, y, unit, team in state_data["agent_map"].values():
            dot_x = map_rect.x + int(x / self.grid_w * map_w)
            dot_y = map_rect.y + int(y / self.grid_h * map_h)
            color = COLORS["red"] if team == 2.0 else COLORS["blue"]
            pygame.draw.rect(surf, color, (dot_x, dot_y, 2, 2))

        # Draw camera view frustum
        if self.viewer.cam.cell_px > 0:
            cam_rect_world = pygame.Rect(self.viewer.cam.offset_x, self.viewer.cam.offset_y,
                                        self.viewer.layout.world_rect().width / self.viewer.cam.cell_px,
                                        self.viewer.layout.world_rect().height / self.viewer.cam.cell_px)
            
            cam_rect_map = pygame.Rect(
                map_rect.x + (cam_rect_world.x / self.grid_w * map_w),
                map_rect.y + (cam_rect_world.y / self.grid_h * map_h),
                (cam_rect_world.width / self.grid_w * map_w),
                (cam_rect_world.height / self.grid_h * map_h)
            )
            pygame.draw.rect(surf, COLORS["marker"], cam_rect_map, 1)

        pygame.draw.rect(surf, COLORS["border"], map_rect, 1)


class InputHandler:
    def __init__(self, viewer): self.viewer = viewer
    def handle(self):
        running, advance_tick = True, False
        keys = pygame.key.get_pressed()
        pan_speed = 10.0 / max(1.0, float(self.viewer.cam.cell_px))
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: self.viewer.cam.pan(-pan_speed, 0)
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: self.viewer.cam.pan(pan_speed, 0)
        if keys[pygame.K_w] or keys[pygame.K_UP]: self.viewer.cam.pan(0, -pan_speed)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: self.viewer.cam.pan(0, pan_speed)
        
        wrect = self.viewer.layout.world_rect()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: running = False
            elif ev.type == pygame.VIDEORESIZE:
                if not self.viewer.fullscreen:
                    self.viewer.Wpix, self.viewer.Hpix = max(800, ev.w), max(520, ev.h)
                    self.viewer.screen = pygame.display.set_mode((self.viewer.Wpix, self.viewer.Hpix), pygame.RESIZABLE)
                    self.viewer.world_renderer.static_surf = None
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE: running = False
                elif ev.key == pygame.K_SPACE: self.viewer.paused = not self.viewer.paused
                elif ev.key == pygame.K_PERIOD and self.viewer.paused: advance_tick = True
                elif ev.key == pygame.K_m and self.viewer.selected_id is not None:
                    if self.viewer.selected_id in self.viewer.marked: self.viewer.marked.remove(self.viewer.selected_id)
                    elif len(self.viewer.marked) < 10: self.viewer.marked.append(self.viewer.selected_id)
                elif ev.key == pygame.K_r: self.viewer.show_rays = not self.viewer.show_rays
                elif ev.key == pygame.K_t: self.viewer.threat_vision_mode = not self.viewer.threat_vision_mode
                elif ev.key == pygame.K_b: self.viewer.battle_view_enabled = not self.viewer.battle_view_enabled
                elif ev.key == pygame.K_s: self.viewer.save_selected_brain()
                elif ev.key == pygame.K_F11:
                    self.viewer.fullscreen = not self.viewer.fullscreen
                    if self.viewer.fullscreen:
                        self.viewer.screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
                    else:
                        self.viewer.screen = pygame.display.set_mode((self.viewer.Wpix, self.viewer.Hpix), pygame.RESIZABLE)
                    self.viewer.world_renderer.static_surf = None
                elif ev.key == pygame.K_EQUALS or ev.key == pygame.K_PLUS:
                    self.viewer.speed_multiplier = min(16, self.viewer.speed_multiplier * 2)
                elif ev.key == pygame.K_MINUS:
                    self.viewer.speed_multiplier = max(0.25, self.viewer.speed_multiplier / 2)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1 and wrect.collidepoint(ev.pos):
                    gx, gy = self.viewer.cam.screen_to_world(ev.pos[0] - wrect.x, ev.pos[1] - wrect.y)
                    aid = int(self.viewer.grid[2, gy, gx].item())
                    self.viewer.selected_id = aid if aid >= 0 else None
                elif ev.button == 4: self.viewer.cam.zoom_at(1.12); self.viewer.world_renderer.static_surf = None
                elif ev.button == 5: self.viewer.cam.zoom_at(1/1.12); self.viewer.world_renderer.static_surf = None
        return running, advance_tick

class AnimationManager:
    def __init__(self): self.animations = []
    def add(self, anim_type, pos): self.animations.append([anim_type, pos, 10])
    def update(self): self.animations = [[t,p,l-1] for t,p,l in self.animations if l > 0]
    def draw(self, surf, wrect, cam):
        c = cam.cell_px
        for anim_type, pos, lifetime in self.animations:
            cx, cy = cam.world_to_screen(pos[0], pos[1])
            alpha = int(255 * (lifetime / 10))
            if anim_type == "damage":
                pygame.draw.circle(surf, (*COLORS["red"], alpha), (wrect.x+cx+c//2, wrect.y+cy+c//2), c//2, 2)

class Viewer:
    def __init__(self, grid: torch.Tensor, cell_size: Optional[int] = None, show_grid: bool = True):
        _center_window(); pygame.init()
        pygame.display.set_caption("Codex Bellum - Transformer")
        self.grid, self.margin, self.show_grid = grid, 8, show_grid
        self.cam = Camera(int(cell_size or config.CELL_SIZE), grid.shape[2], grid.shape[1])
        H, W = grid.shape[1], grid.shape[2]
        side_min_w, hud_h = 280, 126
        
        # --- MODIFICATION START ---
        # Get actual screen size to prevent window from being too large
        try:
            display_info = pygame.display.Info()
            max_w = display_info.current_w - 80  # Subtract padding for taskbar/decorations
            max_h = display_info.current_h - 120
        except pygame.error:
            # Fallback if display is not ready (e.g., headless environment)
            max_w, max_h = 1280, 720
        # --- MODIFICATION END ---
        
        world_px_w, world_px_h = W * self.cam.cell_px, H * self.cam.cell_px
        init_w = min(max_w, max(1280, world_px_w + side_min_w + 3 * self.margin))
        init_h = min(max_h, max(720, world_px_h + hud_h + 2 * self.margin))
        
        self.Wpix, self.Hpix = int(init_w), int(init_h)
        self.screen = pygame.display.set_mode((self.Wpix, self.Hpix), pygame.RESIZABLE)
        
        self.text_cache = TextCache()
        self.clock = pygame.time.Clock()
        self.selected_id: Optional[int] = None
        self.marked: List[int] = []
        self.show_rays = False
        self.paused = False
        self.threat_vision_mode = False
        self.battle_view_enabled = False
        self.fullscreen = False
        self.speed_multiplier = 1.0

    def save_selected_brain(self):
        if self.selected_id is None or not hasattr(self, 'registry'): return
        brain = self.registry.brains[self.selected_id]
        if brain:
            tick = self.stats.tick if hasattr(self, 'stats') else 0
            filename = f"brain_agent_{self.selected_id}_t_{tick}.pth"
            try:
                torch.save(brain.state_dict(), filename)
                print(f"[Viewer] Saved brain for agent {self.selected_id} to '{filename}'")
            except Exception as e:
                print(f"[Viewer] Error saving brain: {e}")

    def run(self, engine, registry, stats, tick_limit: int = 0, target_fps: Optional[int] = None):
        self.registry, self.stats = registry, stats
        self.layout = LayoutManager(self)
        self.world_renderer = WorldRenderer(self, self.grid, registry)
        self.hud_panel = HudPanel(self, stats)
        self.side_panel = SidePanel(self, registry)
        self.input_handler = InputHandler(self)
        self.anim_manager = AnimationManager()
        self.minimap = Minimap(self)
        self.world_renderer.build_static_cache(engine)
        
        running = True
        frame_count = 0
        while running:
            running, advance_tick = self.input_handler.handle()
            
            num_ticks_this_frame = 0
            if not self.paused:
                if self.speed_multiplier >= 1:
                    num_ticks_this_frame = int(self.speed_multiplier)
                elif self.speed_multiplier > 0 and frame_count % int(1 / self.speed_multiplier) == 0:
                    num_ticks_this_frame = 1
            elif advance_tick:
                num_ticks_this_frame = 1
            
            for _ in range(num_ticks_this_frame):
                # This logic is kept in case animations are re-enabled later
                # pre_tick_hp = {i: registry.agent_data[i, COL_HP].item() for i in range(registry.capacity) if registry.agent_data[i, COL_ALIVE] > 0.5}
                engine.run_tick()
                # for i, old_hp in pre_tick_hp.items():
                #     if i < registry.capacity and registry.agent_data[i, COL_ALIVE] > 0.5:
                #         new_hp = registry.agent_data[i, COL_HP].item()
                #         if new_hp < old_hp:
                #             pos = (registry.agent_data[i, COL_X].item(), registry.agent_data[i, COL_Y].item())
                #             self.anim_manager.add("damage", pos)
                self.hud_panel.update()

            # self.anim_manager.update() # Removed as requested

            with torch.no_grad():
                grid_cpu = self.grid.detach().cpu()
                alive_mask = registry.agent_data[:, COL_ALIVE] > 0.5
                alive_indices_tensor = torch.nonzero(alive_mask).squeeze(1)
                alive_indices = alive_indices_tensor.cpu().tolist()
            
                agent_map = {i: (
                    registry.agent_data[i, COL_X].item(),
                    registry.agent_data[i, COL_Y].item(),
                    registry.agent_data[i, COL_UNIT].item(),
                    registry.agent_data[i, COL_TEAM].item()
                ) for i in alive_indices}

                state_data = {
                    "occ_np": grid_cpu[0].numpy().astype(np.int16),
                    "id_np": grid_cpu[2].numpy().astype(np.int32),
                    "alive_indices": alive_indices,
                    "agent_map": agent_map
                }
            
            self.screen.fill(COLORS["bg"])
            self.world_renderer.draw(self.screen, state_data)
            self.hud_panel.draw(self.screen, state_data)
            self.side_panel.draw(self.screen, state_data)
            
            pygame.display.flip()
            self.clock.tick(int(target_fps or config.TARGET_FPS))
            frame_count +=1
            if tick_limit > 0 and stats.tick >= tick_limit: running = False
        
        pygame.quit()
