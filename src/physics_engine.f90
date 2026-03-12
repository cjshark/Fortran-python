subroutine verlet_step(pos, vel, acc, dt, n)
    implicit none

    integer, intent(in) :: n
    real(8), intent(in) :: dt
    real(8), intent(inout) :: pos(2, n)
    real(8), intent(inout) :: vel(2, n)
    real(8), intent(in)    :: acc(2, n)
    !f2py integer, intent(in) :: n
    !f2py real(8), intent(in) :: dt
    !f2py real(8), intent(in,out) :: pos(2,n)
    !f2py real(8), intent(in,out) :: vel(2,n)
    !f2py real(8), intent(in) :: acc(2,n)

    integer :: i

    do i = 1, n
        pos(1, i) = pos(1, i) + vel(1, i) * dt + 0.5d0 * acc(1, i) * dt * dt
        pos(2, i) = pos(2, i) + vel(2, i) * dt + 0.5d0 * acc(2, i) * dt * dt
        vel(1, i) = vel(1, i) + acc(1, i) * dt
        vel(2, i) = vel(2, i) + acc(2, i) * dt
    end do

end subroutine verlet_step

subroutine resolve_collisions(pos, vel, n, radi us)
    implicit none
    !f2py intent(in) :: n, radius
    !f2py intent(inout) :: vel
    !f2py intent(in) :: pos
    
    integer, intent(in) :: n
    real(8), intent(in) :: radius
    real(8), intent(in) :: pos(2, n)
    real(8), intent(inout) :: vel(2, n)
    
    integer :: i, j
    real(8) :: dx, dy, dist_sq, min_dist_sq, relative_vel_x, relative_vel_y
    real(8) :: dot_product, nx, ny
    
    min_dist_sq = (2.0_8 * radius)**2

    do i = 1, n
        do j = i + 1, n
            dx = pos(1, j) - pos(1, i)
            dy = pos(2, j) - pos(2, i)
            dist_sq = dx*dx + dy*dy
            
            if (dist_sq < min_dist_sq .and. dist_sq > 0) then
                ! 1. Normal vector between particles
                nx = dx / sqrt(dist_sq)
                ny = dy / sqrt(dist_sq)
                
                ! 2. Relative velocity
                relative_vel_x = vel(1, i) - vel(1, j)
                relative_vel_y = vel(2, i) - vel(2, j)
                
                ! 3. Dot product (how much they are moving toward each other)
                dot_product = relative_vel_x * nx + relative_vel_y * ny
                
                ! 4. Only bounce if they are moving toward each other
                if (dot_product > 0) then
                    vel(1, i) = vel(1, i) - dot_product * nx
                    vel(2, i) = vel(2, i) - dot_product * ny
                    vel(1, j) = vel(1, j) + dot_product * nx
                    vel(2, j) = vel(2, j) + dot_product * ny
                end if
            end if
        end do
    end do
end subroutine resolve_collisions

subroutine apply_constraints(pos, n, rest_len)
    implicit none
    !f2py intent(in) :: n, rest_len
    !f2py intent(inout) :: pos
    
    integer, intent(in) :: n
    real(8), intent(in) :: rest_len
    real(8), intent(inout) :: pos(2, n)
    
    integer :: i
    real(8) :: dx, dy, dist, diff, percent, offset_x, offset_y

    ! We loop from 1 to n-1, connecting particle i to i+1
    do i = 1, n - 1
        dx = pos(1, i+1) - pos(1, i)
        dy = pos(2, i+1) - pos(2, i)
        dist = sqrt(dx*dx + dy*dy)
        
        if (dist > 0) then
            diff = (rest_len - dist) / dist
            
            ! Push/Pull particles toward each other
            ! We multiply by 0.5 because each particle moves half the distance
            offset_x = dx * diff * 0.5_8
            offset_y = dy * diff * 0.5_8
            
            ! Particle 1 (Stay fixed if i=1 for the 'anchor' effect)
            if (i > 1) then
                pos(1, i) = pos(1, i) - offset_x
                pos(2, i) = pos(2, i) - offset_y
            end if
            
            ! Particle 2
            pos(1, i+1) = pos(1, i+1) + offset_x
            pos(2, i+1) = pos(2, i+1) + offset_y
        end if
    end do
end subroutine apply_constraints

subroutine apply_orbital_gravity(pos, acc, n, sun_x, sun_y, gravity_const)
    implicit none
    !f2py intent(in) :: n, sun_x, sun_y, gravity_const
    !f2py intent(in) :: pos
    !f2py intent(inout) :: acc
    
    integer, intent(in) :: n
    real(8), intent(in) :: sun_x, sun_y, gravity_const
    real(8), intent(in) :: pos(2, n)
    real(8), intent(inout) :: acc(2, n)
    
    integer :: i
    real(8) :: dx, dy, dist_sq, dist, force_mag

    do i = 1, n
        dx = sun_x - pos(1, i)
        dy = sun_y - pos(2, i)
        dist_sq = dx*dx + dy*dy
        
        if (dist_sq > 100.0_8) then ! Prevent "division by zero" explosion
            dist = sqrt(dist_sq)
            ! Newtonian Gravity: F = G / r^2
            force_mag = gravity_const / dist_sq
            
            ! Directional acceleration
            acc(1, i) = force_mag * (dx / dist)
            acc(2, i) = force_mag * (dy / dist)
        else
            acc(1, i) = 0.0_8
            acc(2, i) = 0.0_8
        end if
    end do
end subroutine apply_orbital_gravity