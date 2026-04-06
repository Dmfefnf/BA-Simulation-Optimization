"""
Module providing basic ingredients for animating discrete processes using Salabim.
"""

import salabim as sim
from typing import Callable, Tuple, Union
from math import sqrt
import inspect

ZOOM = 1
BASE_UNIT = 20 * ZOOM

FONT = "Calibri"
FONT_SIZE = BASE_UNIT
TEXT_COLOR = "bg"
LABEL_COLOR = "black"

ENTITY_CREATION_LOC_X = 0
ENTITY_CREATION_LOC_Y = 0
ENTITY_WIDTH = BASE_UNIT * 4
ENTITY_HEIGHT = BASE_UNIT
ENTITY_COLOR = "royalblue"
ENTITY_SPEED = 100

STATION_WIDTH = BASE_UNIT * 5
STATION_HEIGHT = BASE_UNIT * 5
STATION_COLOR = "chocolate"
STATION_TEXT_ANCHOR = "nw"
STATION_LABEL_INDENT = STATION_WIDTH / 10
STATION_LABEL_OFFSET = STATION_HEIGHT / 5
STATION_QUEUE_OFFSET = STATION_HEIGHT / 5
STATION_QUEUE_DIRECTION = "n"


TANK_WIDTH = STATION_WIDTH * 2 / 3
TANK_HEIGHT = STATION_HEIGHT


def get_kwargs_of_class(cls, kwargs: dict) -> dict:
    """
    Get the kwargs that are relevant for a class.
    """
    sig = inspect.signature(sim.Resource.__init__)
    return {
        name: kwargs[name]
        for name, param in sig.parameters.items()
        if param.kind in {param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD}
        and name != "self"
        and param.kind != param.VAR_KEYWORD  # Exclude **kwargs
        and name in kwargs
    }


class BasicEntity(sim.Component):
    """
    Basic entity component with a graphic representation as rectangle and text.
    """

    def setup(
        self,
        x: float = ENTITY_CREATION_LOC_X,
        y: float = ENTITY_CREATION_LOC_Y,
        width: float = ENTITY_WIDTH,
        height: float = ENTITY_HEIGHT,
        fillcolor: str = ENTITY_COLOR,
        textcolor: str = TEXT_COLOR,
        font: str = FONT,
        fontsize: float = FONT_SIZE,
        speed: float = ENTITY_SPEED,
        visible: bool = True,
    ):
        assert width >= 0
        assert height >= 0
        assert fontsize >= 0
        assert speed > 0
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.speed = speed
        self.anim_rect = sim.Animate(
            rectangle0=(self.x, self.y, self.x + self.width, self.y + self.height),
            fillcolor0=fillcolor,
            visible=visible,
            layer=0,
            parent=self,
        )
        self.anim_text = sim.Animate(
            x0=self.x + self.width / 2,
            y0=self.y + self.height / 2,
            text=self.name(),
            textcolor0=textcolor,
            font=font,
            fontsize0=fontsize,
            visible=visible,
            layer=-1,
            parent=self,
        )

    def visible(self, visible: bool = True):
        self.anim_rect.update(visible=visible)
        self.anim_text.update(visible=visible)

    def invisible(self):
        self.visible(False)

    def update_fillcolor(self, fillcolor, duration: Union[float, Callable] = None):
        duration = 0 if duration is None else self.env.spec_to_duration(duration)
        t1 = self.env.now() + max(0, duration)
        self.anim_rect.update(fillcolor1=fillcolor, t1=t1)

    def move(self, x1: float, y1: float, duration: float = None):
        """
        Move the entity to new coordinates (uniform motion on straight line)
        without self.hold().
        If duration=None, use speed and distance to compute duration.
        Use this to move an entity from within another process.
        """
        self.x = self.anim_rect.x()
        self.y = self.anim_rect.y()
        if duration is None:
            duration = sqrt((x1 - self.x) ** 2 + (y1 - self.y) ** 2) / self.speed
        else:
            duration = self.env.spec_to_duration(duration)
        t1 = self.env.now() + max(0, duration)
        self.anim_rect.update(
            rectangle1=(x1, y1, x1 + self.width, y1 + self.height),
            t1=t1,
        )
        self.anim_text.update(
            x1=x1 + self.width / 2,
            y1=y1 + self.height / 2,
            t1=t1,
        )
        self.x = x1
        self.y = y1

    def move_and_hold(
        self,
        x1: float,
        y1: float,
        duration: Union[float, Callable] = None,
        mode: str = None,
    ) -> None:
        """
        Move the entity to new coordinates (uniform motion on straight line)
        without self.hold().
        If duration=None, use speed and distance to compute duration.
        Use this to move an entity from within its own process, holding it for
        the duration of the motion.
        """
        self.x = self.anim_rect.x()
        self.y = self.anim_rect.y()
        if duration is None:
            duration = sqrt((x1 - self.x) ** 2 + (y1 - self.y) ** 2) / self.speed
        else:
            duration = self.env.spec_to_duration(duration)
        t1 = self.env.now() + max(0, duration)
        self.anim_rect.update(
            rectangle1=(x1, y1, x1 + self.width, y1 + self.height),
            t1=t1,
        )
        self.anim_text.update(
            x1=x1 + self.width / 2,
            y1=y1 + self.height / 2,
            t1=t1,
        )
        self.hold(till=t1, mode=mode)
        self.x = x1
        self.y = y1

    def animation_objects(self, id) -> Tuple[float, float, sim.AnimateRectangle]:
        """
        Return a list of animation objects for this entity. Used by sim.AnimateQueue.
        See documentation of sim.AnimateQueue for details.
        TODO: Maybe we can return the animation objects directly?
        """
        ao0 = sim.AnimateRectangle(
            text=self.name(),
            textcolor=self.anim_text.textcolor(),
            font=self.anim_text.font(),
            fontsize=self.anim_text.fontsize(),
            spec=(0, 0, self.width, self.height),
            linewidth=0,
            fillcolor=self.anim_rect.fillcolor(),
            parent=self,
        )
        return (self.width * 1.1, self.height * 1.1, ao0)


class BasicStation:
    """
    Basic station with a graphic representation as rectangle and text.
    This class is not a Salabim component and therefore cannot be used as a process.
    It should be used as a base class for other station classes.
    """

    def __init__(
        self,
        display_name: str = "station",
        x: float = 0,
        y: float = 0,
        width: float = STATION_WIDTH,
        height: float = STATION_HEIGHT,
        fillcolor: str = STATION_COLOR,
        textcolor: str = TEXT_COLOR,
        font: str = FONT,
        fontsize: float = FONT_SIZE,
        text_anchor: str = STATION_TEXT_ANCHOR,
        label_color: str = LABEL_COLOR,
        label_indent: float = STATION_LABEL_INDENT,
        label_offset: float = STATION_LABEL_OFFSET,
        **kwargs,
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.display_name = display_name
        self.anim_background = sim.AnimateRectangle(
            spec=(
                self.x,
                self.y,
                self.x + self.width,
                self.y + self.height,
            ),
            fillcolor=fillcolor,
            text=display_name,
            font=font,
            fontsize=fontsize,
            textcolor=textcolor,
            text_anchor=text_anchor,
        )
        self.anim_label = sim.AnimateText(
            text=lambda _: self.label(),
            x=self.x + label_indent,
            y=self.y - label_offset,
            textcolor=label_color,
            font=font,
            fontsize=fontsize,
            text_anchor=text_anchor,
        )

    def label(self) -> str:
        return ""


class CounterStation(BasicStation):
    """
    A station with a counter and a graphic representation as rectangle
    with a label that shows the current counter value.
    The station is not a Salabim component and therefore has no process.
    The counter is a Salabim state, which has a monitor and its value can be waited for.
    """

    def __init__(self, **station_kwargs):
        BasicStation.__init__(self, **station_kwargs)
        self.count = sim.State(
            name=self.display_name + ".count",
            value=0,
            type="int64",
        )

    def label(self) -> str:
        return f"count: {self.count()}"

    def inc_count(self, step: int = 1) -> None:
        self.count.set(self.count() + step)

    def dec_count(self, step: int = 1) -> None:
        self.count.set(self.count() - step)

    def reset_count(self, value: int = 0) -> None:
        self.count.set(value)


class QueueStation(sim.Queue, BasicStation):
    """
    A station that is a Salabim queue and has a graphic representation as rectangle
    with a label that shows the total number of arrivals and the current queue length.
    The station is not a Salabim component and therefore has no process.
    """

    def __init__(
        self,
        queue_animate: bool = True,
        queue_max_length: int = None,
        queue_direction: str = STATION_QUEUE_DIRECTION,
        queue_offset: float = STATION_QUEUE_OFFSET,
        **kwargs,
    ):
        # Remove kwargs not relevant for sim.Resource to avoid errors.
        queue_kwargs = get_kwargs_of_class(sim.Queue, kwargs)
        sim.Queue.__init__(self, **queue_kwargs)
        BasicStation.__init__(self, **kwargs)
        self.anim_queue = (
            sim.AnimateQueue(
                self,
                x=self.x,
                y=self.y + self.height + queue_offset,
                direction=queue_direction,
                max_length=queue_max_length,
                title="",
                # title=lambda t: str(len(self.queue)), # TODO
                # titlecolor=mm.LABEL_COLOR,
                # titlefont=mm.FONT,
                # titlefontsize=mm.FONT_SIZE,
            )
            if queue_animate
            else None
        )

    def label(self) -> str:
        return f"count: {self.number_of_arrivals}\nqueue: {len(self)}"


class ResourceStation(sim.Resource, BasicStation):
    """
    A station that is a Salabim resource and has a graphic representation as rectangle
    with a label that shows the total number of processed entities and the current number of entities in the resource queue.
    The station is not a Salabim component and therefore has no process.
    """

    def __init__(
        self,
        queue_animate=True,
        queue_max_length=None,
        queue_offset=STATION_QUEUE_OFFSET,
        queue_direction=STATION_QUEUE_DIRECTION,
        **kwargs,
    ):
        # Remove kwargs not relevant for sim.Resource to avoid errors.
        resource_kwargs = get_kwargs_of_class(sim.Resource, kwargs)
        sim.Resource.__init__(self, **resource_kwargs)
        BasicStation.__init__(self, **kwargs)
        # Neu: State der verfügbaren Kapazität, damit andere darauf warten können
        self._available_quantity_state = sim.State(
            name=self.name() + ".available_quantity",
            value=self.capacity(),
        )
        self.anim_queue = (
            sim.AnimateQueue(
                self.requesters(),
                x=self.x,
                y=self.y + self.height + queue_offset,
                direction=queue_direction,
                max_length=queue_max_length,
                title="",
                # title=lambda t: str(len(self.requesters())),  # TODO
                # titlecolor=mm.LABEL_COLOR,
                # titlefont=mm.FONT,
                # titlefontsize=mm.FONT_SIZE,
            )
            if queue_animate
            else None
        )

    def _update_available_state(self):
        self._available_quantity_state.set(self.available_quantity())

    # Salabim ruft diese Hooks automatisch auf:
    def claimed(self, quantity, claimer):
        super().claimed(quantity, claimer)
        self._update_available_state()

    def released(self, quantity, releaser):
        super().released(quantity, releaser)
        self._update_available_state()

    def label(self) -> str:
        return f"cap: {self.claimed_quantity()}/{int(self.capacity())}\nqueue: {len(self.requesters())}\ndone: {self.claimers().number_of_departures}"

    @property
    def available_quantity_state(self):
        return self._available_quantity_state

    